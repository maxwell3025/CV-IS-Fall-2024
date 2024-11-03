from abc import abstractmethod

class LanguageSelectTask:
    """Represents a set of languages indexed by integers.
    """

    @abstractmethod
    def sample(self, length: int, type: int) -> list[int] | None:
        """Sample an instance of this task.

        Args:
            len: The length of the instance.
            type: The index of the language we want to sample from.
        Returns:
            Either a list of integers representing our string, or None if the
            underlying algorithm fails to generate as string of length `length`
        """
        pass

    @abstractmethod
    def repr_sample(self, length: int, type: int) -> str | None:
        """Sample an instance of this task and represent it as a string.

        Args:
            len: The length of the instance.
            type: The index of the language we want to sample from.
        Returns:
            Either a string of length `length` in the chosen language, or None
            if the underlying algorithm fails to generate the string.
        """
        pass

    @abstractmethod
    def alphabet_size(self) -> int:
        """Returns the size of the alphabet for this task.
        """
        pass

    @abstractmethod
    def language_count(self) -> int:
        """Returns the number of languages in this task."""
