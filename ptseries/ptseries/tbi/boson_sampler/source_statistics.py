from scipy.optimize import fsolve


class HeraldedSource:
    """Class for generating the photon statistics of a multiplexed photon source.

    Args:
        detector_num: number of multiplexed detectors on the herald arm.
        herald_efficiency: the efficiency on the herald arm
        dark count: the number of dark counts per second
        p_coefficient: related to the pair gen prob by p_n = (1 - p)p**n. parameterized by the squeezing parameter, r.
    """

    def __init__(self, detector_num: int, herald_efficiency: float, p_coefficient: float, dark_count: float = 0.0):
        self.detector_num = detector_num
        self.herald_efficiency = herald_efficiency
        self.dark_count = dark_count
        self.p_coefficient = p_coefficient

    @property
    def herald_efficiency_bar(self) -> float:
        return 1 - self.herald_efficiency

    @property
    def dark_count_bar(self) -> float:
        return 1 - self.dark_count

    @property
    def p_coefficient_bar(self) -> float:
        return 1 - self.p_coefficient

    @property
    def herald_eff_transform(self) -> float:
        return self.herald_efficiency_bar + (self.herald_efficiency / self.detector_num)

    def prob_herald(self) -> float:
        """Calculates the probability of a herald detection event, equation (7) in notes.

        Returns:
            probability between 0.0 and 1.0.
        """
        prob = (
            self.detector_num
            * self.dark_count_bar ** (self.detector_num - 1)
            * self.p_coefficient_bar
            * (
                (1 / (1 - self.p_coefficient * self.herald_eff_transform))
                - (self.dark_count_bar / (1 - self.p_coefficient * self.herald_efficiency_bar))
            )
        )
        return prob

    def prob_n_given_herald(self, n: int, eps: float = 1e-6) -> float:
        """The conditional probability of n photons given a herald event, equation (8) in notes .

        Returns:
            probability between 0.0 and 1.0.
        """
        numerator = (
            self.detector_num
            * self.dark_count_bar ** (self.detector_num - 1)
            * (self.herald_eff_transform**n - self.dark_count_bar * self.herald_efficiency_bar**n)
            * self.p_coefficient_bar
            * self.p_coefficient**n
        )
        prob_herald = max(self.prob_herald(), eps)
        prob = numerator / prob_herald

        return prob

    def prob_no_herald_given_n(self, n: int) -> float:
        """The probability of vacuum given n photons are generated (i.e. all are lost).

        Returns:
            probability between 0.0 and 1.0.
        """
        return self.dark_count_bar**n * self.herald_efficiency_bar**n

    def prob_n_given_no_herald(self, n: int) -> float:
        """The probability of n photons in the output when there is no herald event (all herald photons lost).

        Returns:
            probability between 0.0 and 1.0.
        """
        gen_p = self.dark_count_bar * self.herald_efficiency_bar * self.p_coefficient
        return gen_p**n * (1 - gen_p)

    def helper_function(self, n: int, x: float) -> float:
        """This is just a short hand helper function used for used for calculating the g2, see equation (10) in notes.

        Returns:
            float
        """
        num1 = (self.herald_eff_transform * x) ** n
        den1 = (1 - self.herald_eff_transform * x) ** (n + 1)

        num2 = self.dark_count_bar * (self.herald_efficiency_bar * x) ** n
        den2 = (1 - self.herald_efficiency_bar * x) ** (n + 1)

        return (num1 / den1) - (num2 / den2)

    def heralded_auto_correlation(self) -> float:
        """Computes the second order heralded auto correlation function.

        Returns:
            float, value greater than zero.
        """
        num = 2 * self.helper_function(2, self.p_coefficient) * self.helper_function(0, self.p_coefficient)
        den = self.helper_function(1, self.p_coefficient) ** 2

        return num / den

    def get_p_from_g2(self, g2: float) -> float:
        """Finds the p_coefficient needed for a given g2.

        Args:
            g2: the target g2 of the source

        Returns:
            p_coefficient needed to get a target g2
        """

        def g2_temp(p_coeff: float) -> float:
            self.p_coefficient = p_coeff
            return self.heralded_auto_correlation() - g2

        p_val = fsolve(g2_temp, 0.5)

        return p_val[0]

    def set_heralded_auto_correlation(self, g2: float) -> None:
        """Set the p coefficient based on a g2 value."""
        p_coeff = self.get_p_from_g2(g2)
        self.p_coefficient = p_coeff

        return None
