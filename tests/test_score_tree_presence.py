"""User-specified hierarchical existence matching; no one-to-one consumption."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/score_tree_presence.py'


class PresenceTests(unittest.TestCase):
    def scorer(self):
        self.assertTrue(SCRIPT.exists(),'Hierarchical presence scorer is missing')
        spec=importlib.util.spec_from_file_location('presence_cli',SCRIPT)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        return mod

    def event(self, kind='WKS', trigger='show', arg='evidence', role='Content'):
        return {'event_type':kind,'trigger':{'tokens':trigger.split()},
                'arguments':[{'tokens':arg.split(),'argument_type':role}]}

    def test_one_prediction_can_match_repeated_gold_expression(self):
        scorer=self.scorer();e=self.event()
        r=scorer.score_documents([[e,e]],[[e]])
        for key in ('TI','TC','AI','AC'):
            self.assertEqual(r['metrics'][key+'_f1'],1.0)
        self.assertEqual(r['counts']['TI']['gold_matched'],2)
        self.assertEqual(r['counts']['TI']['predicted_matched'],1)

    def test_repeated_trigger_different_types_are_not_overwritten(self):
        scorer=self.scorer();events=[self.event(),self.event(kind='FIN')]
        r=scorer.score_documents([events],[list(reversed(events))])
        self.assertEqual(r['metrics']['TC_f1'],1.0)
        self.assertEqual(r['metrics']['AC_f1'],1.0)

    def test_extra_wrong_prediction_lowers_precision(self):
        scorer=self.scorer();e=self.event()
        r=scorer.score_documents([[e]],[[e,self.event(trigger='claim')]])
        self.assertEqual(r['metrics']['TI_p'],0.5)
        self.assertEqual(r['metrics']['TI_r'],1.0)
        self.assertAlmostEqual(r['metrics']['TI_f1'],2/3)

    def test_all_levels_require_parent_conditions(self):
        scorer=self.scorer();e=self.event()
        r=scorer.score_documents([[e]],[[self.event(kind='FIN')]])
        self.assertEqual(r['metrics']['TI_f1'],1.0)
        for key in ('TC','AI','AC'):self.assertEqual(r['metrics'][key+'_f1'],0.0)
        r=scorer.score_documents([[e]],[[self.event(role='Researcher')]])
        self.assertEqual(r['metrics']['AI_f1'],1.0)
        self.assertEqual(r['metrics']['AC_f1'],0.0)

    def test_raw_tree_pipeline_keeps_reusable_matches(self):
        scorer=self.scorer();e=self.event()
        gold=[{'doc_id':'doc','events':[e,e]}]
        rows=[{'id':'s0','document_id':'doc','sentence_index':0,'sentences':['show evidence']}]
        pred=[{'id':'s0','raw_output':'(ROOT (WKS (Trigger "show") (Content "evidence")))'}]
        result=scorer.score(gold,rows,pred)
        for metric in ('TI','TC','AI','AC'):
            self.assertEqual(result['metrics'][metric+'_f1'],1.0)
        self.assertEqual(result['predicted_events'],1)

    def test_no_matching_across_documents(self):
        scorer=self.scorer();e=self.event()
        r=scorer.score_documents([[e],[]],[[],[e]])
        self.assertEqual(r['metrics']['TI_f1'],0.0)

    def test_subevent_and_argument_multiplicity_can_reuse(self):
        scorer=self.scorer();e=self.event(arg='show')
        r=scorer.score_documents([[e,e]],[[e]])
        self.assertEqual(r['metrics']['EC_C_f1'],1.0)


if __name__=='__main__':unittest.main()
