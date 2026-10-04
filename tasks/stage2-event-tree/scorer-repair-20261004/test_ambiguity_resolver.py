"""Gold-free span enumeration contract and exhaustive small-case oracle."""
import importlib.util
import itertools
from pathlib import Path
import unittest

PATH = Path(__file__).with_name('ambiguity_resolver.py')

class ResolverTests(unittest.TestCase):
    def resolve(self, *args, **kwargs):
        self.assertTrue(PATH.exists(), 'The ambiguity resolver has not been implemented')
        spec = importlib.util.spec_from_file_location('ambiguity_resolver_test', PATH)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module.resolve_span(*args, **kwargs)

    def test_repeated_word_continuation_previously_omitted(self):
        result = self.resolve(['a','b','b','c'], 'a b c')
        self.assertEqual(result['status'], 'ambiguous')
        self.assertEqual(result['candidates'], [[0,1,3],[0,2,3]])
        self.assertIsNone(result['offsets'])

    def test_contiguous_match_does_not_hide_discontinuous(self):
        result = self.resolve(['a','b','a','x','b'], 'a b')
        self.assertEqual(result['candidates'], [[0,1],[0,4],[2,4]])
        self.assertEqual(result['status'], 'ambiguous')

    def test_unique_and_unaligned(self):
        self.assertEqual(self.resolve(['a','x','b'], 'a b')['offsets'], [0,2])
        self.assertEqual(self.resolve(['a'], 'b')['status'], 'unaligned')
        self.assertEqual(self.resolve(['a'], '')['candidates'], [])

    def test_truncation_cannot_be_mislabeled_unique(self):
        result = self.resolve(['a','a'], 'a', max_candidates=1)
        self.assertEqual(result['status'], 'truncated')
        self.assertEqual(result['candidates'], [[0]])
        self.assertTrue(result['candidate_limit_reached'])
        self.assertIsNone(result['offsets'])
        exact_limit = self.resolve(['a','a'], 'a', max_candidates=2)
        self.assertEqual(exact_limit['status'], 'ambiguous')
        self.assertFalse(exact_limit['candidate_limit_reached'])
        self.assertEqual(self.resolve(['a'], 'a', max_candidates=1)['status'], 'unique')

    def test_scope_is_global_and_excludes_outside_matches(self):
        result = self.resolve(['a','b','a','b'], 'a b', scope=(2,4))
        self.assertEqual(result['status'], 'unique')
        self.assertEqual(result['offsets'], [2,3])
        self.assertEqual(self.resolve(['a'], 'a', scope=(1,1))['status'], 'unaligned')

    def test_explicit_reverse_and_duplicate_indices(self):
        reverse = self.resolve(['a','b'], 'b a', explicit_offsets=[1,0])
        self.assertEqual(reverse['status'], 'explicit')
        self.assertEqual(reverse['offsets'], [1,0])
        duplicate = self.resolve(['a','b'], 'a a', explicit_offsets=[0,0])
        self.assertEqual(duplicate['offsets'], [0,0])
        self.assertFalse(duplicate['candidate_limit_reached'])

    def test_empty_explicit_span_is_invalid_but_empty_text_is_unaligned(self):
        with self.assertRaises(ValueError):
            self.resolve(['a'], '', explicit_offsets=[])
        self.assertEqual(self.resolve(['a'], '')['status'], 'unaligned')

    def test_explicit_and_bounds_validation(self):
        for kwargs in ({'explicit_offsets':[0]}, {'explicit_offsets':[2,0]},
                       {'explicit_offsets':[True,0]}, {'explicit_offsets':[1,0], 'scope':(1,2)}):
            with self.assertRaises(ValueError): self.resolve(['a','b'], 'b a', **kwargs)
        for scope in ((-1,2),(0,3),(2,1)):
            with self.assertRaises(ValueError): self.resolve(['a','b'], 'a', scope=scope)
        with self.assertRaises(ValueError): self.resolve(['a'], 'a', max_candidates=0)

    def test_against_independent_combination_oracle(self):
        for n in range(5):
            for tokens in itertools.product('ab', repeat=n):
                for query_size in range(1,4):
                    for query in itertools.product('ab', repeat=query_size):
                        expected = [list(indices) for indices in itertools.combinations(range(n), query_size)
                                    if tuple(tokens[i] for i in indices)==query]
                        result = self.resolve(list(tokens), ' '.join(query), max_candidates=100)
                        self.assertEqual(result['candidates'], expected)
                        self.assertEqual(result['status'], 'unique' if len(expected)==1 else 'ambiguous' if expected else 'unaligned')

    def test_long_unique_span_and_impossible_suffix(self):
        tokens = [str(i) for i in range(1200)]
        result = self.resolve(tokens, ' '.join(tokens))
        self.assertEqual(result['status'], 'unique')
        self.assertEqual(result['offsets'], list(range(1200)))
        self.assertEqual(self.resolve(['a']*1000+['b'], 'b a')['status'], 'unaligned')

if __name__ == '__main__': unittest.main()
