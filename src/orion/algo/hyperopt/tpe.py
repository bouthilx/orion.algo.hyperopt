# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.hyperopt.tpe -- TODO 
================================================

.. module:: tpe
    :platform: Unix
    :synopsis: TODO

TODO: Write long description
"""
from orion.algo.base import BaseAlgorithm

import numpy
try:
    import hyperopt as hpo
    from hyperopt.fmin import generate_trials_to_calculate
except Exception:
    hpo = None


def convert_orion_space_to_hyperopt_space(orion_space):
    """Convert Oríon's definition of problem's domain to a skopt compatible."""
    dimensions = []
    for key, dimension in orion_space.items():
        #  low = dimension._args[0]
        #  high = low + dimension._args[1]
        low, high = dimension.interval()
        # NOTE: A hack, because orion priors have non-inclusive higher bound
        #       while scikit-optimizer have inclusive ones.
        high = numpy.nextafter(high, high - 1)
        shape = dimension.shape
        assert not shape or len(shape) == 1
        if not shape:
            shape = (1,)
        # Unpack dimension
        for i in range(shape[0]):
            dimensions.append(Real(name=key + '_' + str(i),
                                   prior='uniform',
                                   low=low, high=high))

    return Space(dimensions)

class TPE(BaseAlgorithm):
    """
    TODO: Class docstring
    """

    requires = 'real'

    def __init__(self,
                 space,
                 # max_concurrent=10, # Only allow max_concurrent = 1 for now
                 # points_to_evaluate=None,
                 # reward_attr="episode_reward_mean",
                 ):
        """
        TODO: init docstring
        """
        max_concurrent=1
        points_to_evaluate=None
        reward_attr="objective",

        assert hpo is not None, "HyperOpt must be installed!"
        assert type(max_concurrent) is int and max_concurrent > 0
        self._max_concurrent = max_concurrent
        self._reward_attr = reward_attr
        self.algo = hpo.tpe.suggest
        self.domain = hpo.Domain(lambda spc: spc, space)
        if points_to_evaluate is None:
            self._hpopt_trials = hpo.Trials()
            self._points_to_evaluate = 0
        else:
            assert type(points_to_evaluate) == list
            self._hpopt_trials = generate_trials_to_calculate(
                points_to_evaluate)
            self._hpopt_trials.refresh()
            self._points_to_evaluate = len(points_to_evaluate)
        self._live_trial_mapping = {}
        self.rstate = np.random.RandomState()
                
        super(TPE, self).__init__(space)
        # self.optimizer = None ??????

    def suggest(self, num=1):
        new_ids = self._hpopt_trials.new_trial_ids(1)
        self._hpopt_trials.refresh()

        # Get new suggestion from Hyperopt
        new_trials = self.algo(new_ids, self.domain, self._hpopt_trials,
                               self.rstate.randint(2**31 - 1))
        self._hpopt_trials.insert_trial_docs(new_trials)
        self._hpopt_trials.refresh()
        new_trial = new_trials[0]
        self._live_trial_mapping[trial_id] = (new_trial["tid"], new_trial)

        # Taken from HyperOpt.base.evaluate
        config = hpo.base.spec_from_misc(new_trial["misc"])
        ctrl = hpo.base.Ctrl(self._hpopt_trials, current_trial=new_trial)
        memo = self.domain.memo_from_config(config)
        hpo.utils.use_obj_for_literal_in_memo(self.domain.expr, ctrl,
                                              hpo.base.Ctrl, memo)

        suggested_config = hpo.pyll.rec_eval(
            self.domain.expr,
            memo=memo,
            print_node_on_error=self.domain.rec_eval_print_node_on_error)
        # return copy.deepcopy(suggested_config)
        point = copy.deepcopy(suggested_config)
        return [pack_point(point, self.space)]

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        corresponds to "on_trial_result(self, trial_id, result)"

        """
        # unpack_point(points, self.space)[0]
        trial_id = 0
        self.on_trial_complete(trial_id, result=[results[0]['objective']])


    @property
    def is_done(self):
        """Return True, if an algorithm holds that there can be no further improvement."""
        # NOTE: Drop if not used by algorithm
        raise NotImplementedError

    def score(self, point):
        """Allow algorithm to evaluate `point` based on a prediction about
        this parameter set's performance.
        """
        # NOTE: Drop if not used by algorithm
        raise NotImplementedError

    def judge(self, point, measurements):
        """Inform an algorithm about online `measurements` of a running trial."""
        # NOTE: Drop if not used by algorithm
        raise NotImplementedError

    @property
    def should_suspend(self):
        """Allow algorithm to decide whether a particular running trial is still
        worth to complete its evaluation, based on information provided by the
        `judge` method.

        """
        # NOTE: Drop if not used by algorithm
        raise NotImplementedError

    def on_trial_result(self, trial_id, result):
        ho_trial = self._get_hyperopt_trial(trial_id)
        if ho_trial is None:
            return
        now = hpo.utils.coarse_utcnow()
        ho_trial['book_time'] = now
        ho_trial['refresh_time'] = now

    def on_trial_complete(self,
                          trial_id,
                          result=None,
                          error=False,
                          early_terminated=False):
        """
            Passes the result to HyperOpt unless early terminated or errored.
        """
        ho_trial = self._get_hyperopt_trial(trial_id)
        if ho_trial is None:
            return
        ho_trial['refresh_time'] = hpo.utils.coarse_utcnow()
        if error:
            ho_trial['state'] = hpo.base.JOB_STATE_ERROR
            ho_trial['misc']['error'] = (str(TuneError), "Tune Error")
        elif early_terminated:
            ho_trial['state'] = hpo.base.JOB_STATE_ERROR
            ho_trial['misc']['error'] = (str(TuneError), "Tune Removed")
        else:
            ho_trial['state'] = hpo.base.JOB_STATE_DONE
            hp_result = self._to_hyperopt_result(result)
            ho_trial['result'] = hp_result
        self._hpopt_trials.refresh()
        del self._live_trial_mapping[trial_id]

    def _to_hyperopt_result(self, result):
        return {"loss": result[self._reward_attr], "status": "ok"}

    def _get_hyperopt_trial(self, trial_id):
        if trial_id not in self._live_trial_mapping:
            return
        hyperopt_tid = self._live_trial_mapping[trial_id][0]
        return [
            t for t in self._hpopt_trials.trials if t["tid"] == hyperopt_tid
        ][0]

    def _num_live_trials(self):
        return len(self._live_trial_mapping)
