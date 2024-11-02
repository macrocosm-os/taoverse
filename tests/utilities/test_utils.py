import dataclasses
import functools
import time
import unittest
from tempfile import NamedTemporaryFile

from numpy import str_

from taoverse.utilities import utils
from taoverse.utilities.utils import run_in_subprocess, run_in_thread


@dataclasses.dataclass
class SimpleObject:
    int_field: int
    bool_field: bool
    str_field: str
    list_field: list
    dict_field: dict


class TestUtils(unittest.TestCase):
    def test_run_in_subprocess(self):
        def test_func(a: int, b: int):
            return a + b

        partial = functools.partial(test_func, 1, 2)

        result = run_in_subprocess(func=partial, ttl=5)
        self.assertEqual(3, result)

    def test_run_in_subprocess_timeout(self):
        def test_func(a: int, b: int):
            time.sleep(3)
            return a + b

        partial = functools.partial(test_func, 1, 2)

        with self.assertRaises(TimeoutError):
            result = run_in_subprocess(func=partial, ttl=1)

    def test_run_in_subprocess_no_return(self):
        def test_func(a: int, b: int):
            pass

        partial = functools.partial(test_func, 1, 2)

        result = run_in_subprocess(func=partial, ttl=5)
        self.assertIsNone(result)

    def test_run_in_subprocess_tuple_return(self):
        def test_func(a: int, b: int):
            return a, b

        partial = functools.partial(test_func, 1, 2)

        result = run_in_subprocess(func=partial, ttl=5)
        self.assertEqual((1, 2), result)

    def test_run_in_subprocess_exception(self):
        def test_func(a: int, b: int):
            raise ValueError()

        partial = functools.partial(test_func, 1, 2)

        with self.assertRaises(ValueError):
            result = run_in_subprocess(func=partial, ttl=5)

    def test_run_in_thread(self):
        def test_func(a: int, b: int):
            return a + b

        partial = functools.partial(test_func, 1, 2)

        result = run_in_thread(func=partial, ttl=5)
        self.assertEqual(3, result)

    def test_run_in_thread_timeout(self):
        def test_func(a: int, b: int):
            time.sleep(3)
            return a + b

        partial = functools.partial(test_func, 1, 2)

        with self.assertRaises(TimeoutError):
            result = run_in_thread(func=partial, ttl=1)

    def test_run_in_thread_no_return(self):
        def test_func(a: int, b: int):
            pass

        partial = functools.partial(test_func, 1, 2)

        result = run_in_thread(func=partial, ttl=5)
        self.assertIsNone(result)

    def test_run_in_thread_tuple_return(self):
        def test_func(a: int, b: int):
            return a, b

        partial = functools.partial(test_func, 1, 2)

        result = run_in_thread(func=partial, ttl=5)
        self.assertEqual((1, 2), result)

    def test_run_in_thread_exception(self):
        def test_func(a: int, b: int):
            raise ValueError()

        partial = functools.partial(test_func, 1, 2)

        with self.assertRaises(ValueError):
            result = run_in_thread(func=partial, ttl=5)

    def test_save_and_load_version(self):
        version = 1023  # Typically equivalent to "1.2.3" in our convention.
        with NamedTemporaryFile() as f:
            self.assertIsNone(utils.get_version(f.name))

            utils.save_version(f.name, version)
            self.assertEqual(utils.get_version(f.name), version)

    def test_fingerprint_single_object(self):
        a = SimpleObject(1, True, "hello", [1, 2, 3], {"a": 1, "b": 2})
        self.assertEqual(utils.fingerprint(a), utils.fingerprint(a))

        b = SimpleObject(1, True, "other", [1, 2, 3], {"a": 1, "b": 2})
        self.assertNotEqual(utils.fingerprint(a), utils.fingerprint(b))

    def test_fingerprint_multiple_objects(self):
        a = SimpleObject(1, True, "hello", [1, 2, 3], {"a": 1, "b": 2})
        b = SimpleObject(1, True, "other", [1, 2, 3], {"a": 1, "b": 2})

        self.assertEqual(utils.fingerprint([a, b]), utils.fingerprint([a, b]))
        self.assertEqual(utils.fingerprint([a]), utils.fingerprint([a]))

        self.assertNotEqual(utils.fingerprint([a]), utils.fingerprint([b]))
        self.assertNotEqual(utils.fingerprint([a, b]), utils.fingerprint([b, a]))


if __name__ == "__main__":
    unittest.main()
