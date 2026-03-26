#!/usr/bin/env python3

from typing import Optional

import numpy as np


# Represents single axis polynomial trajectory
class PolynomialTrajectory(object):

    def __init__(
        self,
        coefficients: Optional[np.ndarray] = None,
        T: Optional[float] = None,
    ) -> None:
        """Stores coefficients in the following form:
        p(t) = coefficients[0] + coefficients[1] * t + ... +
               coefficients[n-1]*t^(n-1)

        Args:
            coefficients: 1xn numpy array representing the coefficients
                          for a single axis trajectory
            T: scalar input for duration of the trajectory
        """

        self.coefficients: Optional[np.ndarray] = coefficients
        self.T: Optional[float] = T

    def derivative(self, order: int = 0) -> np.ndarray:
        """returns the derivative of the coefficients specified by the given order.
        For example:
            - order 0, return same coefficients
            - order 1, return first derivative
            - order 2, return second derivative
            - ...and so on

        Args:
            order: scalar input representing desired derivative for the coefficients

        Output:
            coeffs: 1 x (n-order) numpy array of coefficients, which may
                    be evaluated to obtain the values of the derivative
                    equal to order.
        """

        c = np.asarray(self.coefficients, dtype=float).copy()
        for _ in range(order):
            n = len(c)
            if n <= 1:
                c = np.array([0.0])
            else:
                c = np.array([(i + 1) * c[i + 1] for i in range(n - 1)])
        return c

    def evaluate(self, time: float, order: int) -> float:
        """Takes the derivative of the coefficients specified by the input
            `order` and then evaluates them for time `time`.

        Args:
            order: order of the derivative to take of the coefficients
            time: time at which the derivative of the coefficients are evaluated

        Output:
            value: scalar value representing the evaluation of the polynomial
                   coefficients taken at time time.
        """

        coeffs = self.derivative(order)
        return float(np.polyval(coeffs[::-1], time))

    def get_ref(self, time: float, order: int) -> np.ndarray:
        """Returns the references up to the derivative specified
            by order for the time specified by time.

        Example: get_ref(1, 4) will return a 1x5 np array consisting
                 of position, velocity, acceleration, jerk, snap

        Args:
            time: time at which to evaluate the polynomial up to the
                  derivative specified by order
            order: derivative to evaluate the polynomial
        """

        ref = np.zeros(order + 1)
        for k in range(order + 1):
            ref[k] = self.evaluate(time, k)
        return ref
