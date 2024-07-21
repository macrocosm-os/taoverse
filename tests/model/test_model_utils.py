import unittest

from taoverse.model import utils


class TestModelUtils(unittest.TestCase):
    def test_get_hash_of_two_strings(self):
        string1 = "hello"
        string2 = "world"

        result = utils.get_hash_of_two_strings(string1, string2)

        self.assertEqual(result, "k2oYXKqiZrucvpgengXLeM1zKwsygOuURBK7b4+PB68=")

    def test_validate_hf_repo_id_too_long(self):
        with self.assertRaises(ValueError) as ve:
            # Max allowed length is 41 characters
            utils.validate_hf_repo_id("my-org/" + "a" * 40)

        self.assertRegex(
            str(ve.exception),
            "Hugging Face repo id must be between 3 and 38 characters",
        )

    def test_validate_hf_repo_id_incorrect_format(self):
        with self.assertRaises(ValueError) as ve:
            utils.validate_hf_repo_id("my-repo-name-without-a-namespace")

        self.assertRegex(
            str(ve.exception), "must be in the format <org or user name>/<repo_name>"
        )

    def test_validate_hf_repo_id_valid(self):
        namespace, name = utils.validate_hf_repo_id("my-org/my-repo-name")
        self.assertEqual("my-org", namespace)
        self.assertEqual("my-repo-name", name)


if __name__ == "__main__":
    unittest.main()
