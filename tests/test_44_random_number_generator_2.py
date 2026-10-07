# -*- coding: utf-8 -*-
import time
import unittest
import cylag
import matplotlib.pyplot as plt
import numpy as np

PRINTOUT = False

class CheckRNG(unittest.TestCase):

    def test_random_pair_equals_two_singles_rectangular(self):
        """A generated pair should equal two consecutive single draws."""

        seed = 123456
        stream = 789

        # Generate two values using generate_random_pair().
        cylag.pcg32_seed(seed, stream)
        value_1_pair, value_2_pair = cylag.generate_random_pair(1)

        # Reset the generator to exactly the same initial state.
        cylag.pcg32_seed(seed, stream)

        # Generate the same two values individually.
        value_1_single = cylag.generate_random_single(1)
        value_2_single = cylag.generate_random_single(1)

        self.assertEqual(value_1_pair, value_1_single)
        self.assertEqual(value_2_pair, value_2_single)

    def test_random_pair_equals_two_singles_ziggurat(self):
        """A generated pair should equal two consecutive single draws."""

        seed = 123456
        stream = 789

        # Generate two values using generate_random_pair().
        cylag.pcg32_seed(seed, stream)
        value_1_pair, value_2_pair = cylag.generate_random_pair(2)

        # Reset the generator to exactly the same initial state.
        cylag.pcg32_seed(seed, stream)

        # Generate the same two values individually.
        value_1_single = cylag.generate_random_single(2)
        value_2_single = cylag.generate_random_single(2)

        self.assertEqual(value_1_pair, value_1_single)
        self.assertEqual(value_2_pair, value_2_single)

    def test_generate_random_single_performance(self):
        """Measure and plot CPU time for each single-value method."""

        # Number of calls tested for each method.
        number_of_calls = np.array(
            [1000, 10000, 100000])

        # In random_utils:
        # method == 1 selects the rectangular approximation.
        # Any other value selects the Ziggurat method.
        methods = {"Rectangular": 1, "Ziggurat": 2, "Marsaglia": 3}

        number_of_repetitions = 5
        cpu_times = {}

        for method_name, method_number in methods.items():
            method_times = []
            for call_count in number_of_calls:
                repetition_times = []
                for repetition in range(number_of_repetitions):

                    # Use the same initial state for every repetition.
                    cylag.pcg32_seed(123456, 789)
                    start_time = time.process_time()

                    for _ in range(call_count):
                        cylag.generate_random_single(method_number)

                    end_time = time.process_time()
                    repetition_times.append(end_time - start_time)

                # The median is less sensitive to occasional interruptions.
                median_cpu_time = float(np.median(repetition_times))
                method_times.append(median_cpu_time)

                print(
                    f"{method_name:12s} | "
                    f"{call_count:>9,d} calls | "
                    f"{median_cpu_time:.6f} s"
                )

            cpu_times[method_name] = np.asarray(method_times)

        # Plot total CPU time.
        if PRINTOUT:
            plt.figure(figsize=(8, 5))
            for method_name, method_times in cpu_times.items():
                plt.plot(
                    number_of_calls,
                    method_times,
                    marker="o",
                    label=method_name,)
            plt.xlabel("Number of calls")
            plt.ylabel("CPU time (seconds)")
            plt.title("Performance of generate_random_single()")
            plt.show()

        # Sanity checks only. Timing itself is not asserted because it depends
        # on the processor, operating system, compiler, and current workload.
        for method_name, method_times in cpu_times.items():
            self.assertTrue(
                np.all(np.isfinite(method_times)),
                f"{method_name} produced invalid timing results.")
            self.assertTrue(
                np.all(method_times >= 0.0),
                f"{method_name} produced negative timing results.")

    def test_generate_random_pair_performance(self):
        """Measure and plot CPU time for each single-value method."""

        # Number of calls tested for each method.
        number_of_calls = np.array(
            [1000, 10000, 100000])

        # In random_utils:
        # method == 1 selects the rectangular approximation.
        # Any other value selects the Ziggurat method.
        methods = {"Rectangular": 1, "Ziggurat": 2, "Marsaglia": 3}

        number_of_repetitions = 5
        cpu_times = {}

        for method_name, method_number in methods.items():
            method_times = []
            for call_count in number_of_calls:
                repetition_times = []
                for repetition in range(number_of_repetitions):

                    # Use the same initial state for every repetition.
                    cylag.pcg32_seed(123456, 789)
                    start_time = time.process_time()

                    for _ in range(call_count):
                        cylag.generate_random_pair(method_number)

                    end_time = time.process_time()
                    repetition_times.append(end_time - start_time)

                # The median is less sensitive to occasional interruptions.
                median_cpu_time = float(np.median(repetition_times))
                method_times.append(median_cpu_time)

                print(
                    f"{method_name:12s} | "
                    f"{call_count:>9,d} calls | "
                    f"{median_cpu_time:.6f} s"
                )

            cpu_times[method_name] = np.asarray(method_times)

        # Plot total CPU time.
        if PRINTOUT:
            plt.figure(figsize=(8, 5))
            for method_name, method_times in cpu_times.items():
                plt.plot(
                    number_of_calls,
                    method_times,
                    marker="o",
                    label=method_name,)
            plt.xlabel("Number of calls")
            plt.ylabel("CPU time (seconds)")
            plt.title("Performance of generate_random_pair()")
            plt.show()

        # Sanity checks only. Timing itself is not asserted because it depends
        # on the processor, operating system, compiler, and current workload.
        for method_name, method_times in cpu_times.items():
            self.assertTrue(
                np.all(np.isfinite(method_times)),
                f"{method_name} produced invalid timing results.")
            self.assertTrue(
                np.all(method_times >= 0.0),
                f"{method_name} produced negative timing results.")

if __name__ == "__main__":
    unittest.main(verbosity=2)


if __name__ == "__main__":
    unittest.main()
