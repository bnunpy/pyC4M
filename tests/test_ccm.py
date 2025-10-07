import unittest
import numpy as np
import pandas as pd

from pyc4m import CCM, conditional
from pyc4m.cccm import causalized_ccm
from pyc4m.conditional import conditional_ccm


class CCMTests(unittest.TestCase):
    def test_causalized_ccm_perfect_synchrony(self):
        t = np.linspace(0, 2 * np.pi, 200)
        series = np.sin(t)
        result = causalized_ccm(series, series, tau=-1, e_dim=2, num_skip=5)
        self.assertGreater(result.correlation_x, 0.99)
        self.assertGreater(result.correlation_y, 0.99)

    def test_conditional_ccm_directional_effects(self):
        np.random.seed(42)
        n = 300
        t = np.linspace(0, 6 * np.pi, n)
        x = np.sin(t)
        z = np.cos(0.5 * t)
        y = 0.7 * np.roll(x, 1) + 0.3 * z + 0.05 * np.random.randn(n)

        data = np.column_stack([x, y, z])
        result = conditional_ccm(
            data, tau=-1, e_dim=3, pairs=[(0, 1)], num_skip=10, exclusion_radius=2
        )
        pair = result.pair_results[(0, 1)]

        self.assertGreater(pair.x_on_y, pair.y_on_x)
        self.assertGreater(pair.diagnostics["var_x_conditionals"], 0)
        self.assertGreater(pair.diagnostics["var_y_conditionals"], 0)

    def test_api_ccm_compatible_interface(self):
        t = np.linspace(0, 4 * np.pi, 220)
        x = np.sin(t)
        y = np.roll(x, 2)
        frame = pd.DataFrame({"x": x, "y": y})

        result = CCM(
            dataFrame=frame,
            columns="x",
            target="y",
            libSizes=[120, 180],
            E=2,
            tau=-1,
            num_skip=5,
            causal=True,
        )

        self.assertListEqual(list(result.columns), ["LibSize", "x:y", "y:x"])
        self.assertEqual(len(result), 2)

    def test_api_ccm_includeData(self):
        t = np.linspace(0, 4 * np.pi, 180)
        x = np.sin(t)
        y = np.roll(x, 1)
        frame = pd.DataFrame({"x": x, "y": y})

        output = CCM(
            dataFrame=frame,
            columns="x",
            target="y",
            libSizes="120 160 20",
            E=2,
            tau=-1,
            includeData=True,
            num_skip=5,
            causal=True,
        )

        self.assertIn("LibMeans", output)
        self.assertIn("PredictStats1", output)
        self.assertEqual(len(output["PredictStats1"]), 3)
        self.assertTrue((output["PredictStats1"]["Sample"] == 0).all())

    def test_api_ccm_random_sampling_reproducible(self):
        t = np.linspace(0, 6 * np.pi, 320)
        x = np.sin(t)
        y = np.cos(t)
        frame = pd.DataFrame({"x": x, "y": y})

        result_one = CCM(
            dataFrame=frame,
            columns="x",
            target="y",
            libSizes=[180, 220],
            E=3,
            tau=-1,
            sample=4,
            seed=5,
            num_skip=5,
            causal=True,
        )

        result_two = CCM(
            dataFrame=frame,
            columns="x",
            target="y",
            libSizes=[180, 220],
            E=3,
            tau=-1,
            sample=4,
            seed=5,
            num_skip=5,
            causal=True,
        )

        np.testing.assert_allclose(
            result_one[f"x:y"].to_numpy(),
            result_two[f"x:y"].to_numpy(),
            equal_nan=True,
        )

        stats = CCM(
            dataFrame=frame,
            columns="x",
            target="y",
            libSizes=[180, 220],
            E=3,
            tau=-1,
            sample=3,
            seed=2,
            includeData=True,
            num_skip=5,
            causal=True,
        )["PredictStats1"]

        self.assertEqual(stats["Sample"].nunique(), 3)

    def test_api_ccm_exclusion_radius(self):
        t = np.linspace(0, 5 * np.pi, 280)
        x = np.sin(t)
        y = np.roll(x, 4)
        frame = pd.DataFrame({"x": x, "y": y})

        baseline = CCM(
            dataFrame=frame,
            columns="x",
            target="y",
            libSizes=[200],
            E=3,
            tau=-1,
            num_skip=5,
            causal=True,
        )

        excluded = CCM(
            dataFrame=frame,
            columns="x",
            target="y",
            libSizes=[200],
            E=3,
            tau=-1,
            exclusionRadius=12,
            num_skip=5,
            causal=True,
        )

        self.assertEqual(len(baseline), len(excluded))
        self.assertTrue(np.all(np.isfinite(excluded[f"x:y"].to_numpy())))

    def test_conditional_wrapper_exclusion_radius(self):
        t = np.linspace(0, 4 * np.pi, 240)
        x = np.sin(t)
        y = np.roll(x, 2)
        z = np.cos(0.3 * t)
        frame = pd.DataFrame({"x": x, "y": y, "z": z})

        result = conditional(
            dataFrame=frame,
            tau=-1,
            e_dim=3,
            pairs=[(0, 1)],
            num_skip=5,
            exclusionRadius=3,
            causal=True,
        )

        self.assertIn((0, 1), result.pair_results)

    def test_api_ccm_non_causal_option(self):
        t = np.linspace(0, 6 * np.pi, 320)
        x = np.sin(t)
        y = np.roll(x, -4)  # future information present
        frame = pd.DataFrame({"x": x, "y": y})

        causal_result = CCM(
            dataFrame=frame,
            columns="x",
            target="y",
            libSizes=[220],
            E=3,
            tau=-1,
            num_skip=5,
            causal=True,
        )

        non_causal_result = CCM(
            dataFrame=frame,
            columns="x",
            target="y",
            libSizes=[220],
            E=3,
            tau=1,
            num_skip=5,
            causal=False,
        )

        self.assertGreater(
            non_causal_result["x:y"].iloc[-1], causal_result["x:y"].iloc[-1]
        )

    def test_conditional_non_causal_option(self):
        t = np.linspace(0, 6 * np.pi, 320)
        x = np.sin(t)
        y = np.roll(x, -3)
        z = np.cos(0.7 * t)
        frame = pd.DataFrame({"x": x, "y": y, "z": z})

        causal_cond = conditional(
            dataFrame=frame,
            tau=-1,
            e_dim=3,
            pairs=[(0, 1)],
            num_skip=5,
            causal=True,
        )

        non_causal_cond = conditional(
            dataFrame=frame,
            tau=1,
            e_dim=3,
            pairs=[(0, 1)],
            num_skip=5,
            causal=False,
        )

        self.assertNotEqual(
            causal_cond.pair_results[(0, 1)].x_on_y,
            non_causal_cond.pair_results[(0, 1)].x_on_y,
        )

    def test_ccm_causal_requires_negative_tau(self):
        frame = pd.DataFrame({"x": np.sin(np.linspace(0, 2 * np.pi, 120)), "y": np.cos(np.linspace(0, 2 * np.pi, 120))})

        with self.assertRaises(RuntimeError):
            CCM(
                dataFrame=frame,
                columns="x",
                target="y",
                libSizes=[80],
                E=2,
                tau=1,
                causal=True,
            )

    def test_causalized_core_requires_negative_tau(self):
        t = np.linspace(0, 2 * np.pi, 150)
        x = np.sin(t)
        y = np.cos(t)

        with self.assertRaises(ValueError):
            causalized_ccm(x, y, tau=1, e_dim=2, num_skip=5)

    def test_conditional_causal_requires_negative_tau(self):
        t = np.linspace(0, 4 * np.pi, 180)
        data = np.column_stack([
            np.sin(t),
            np.cos(t),
            np.sin(0.5 * t + 0.1),
        ])

        with self.assertRaises(ValueError):
            conditional(
                dataFrame=pd.DataFrame(data, columns=["x", "y", "z"]),
                tau=1,
                e_dim=3,
                pairs=[(0, 1)],
                causal=True,
            )


    def test_causalized_ccm_nonzero_tp(self):
        t = np.linspace(0, 4 * np.pi, 240)
        x = np.sin(t)
        y = np.roll(x, -1)
        result = causalized_ccm(x, y, tau=-1, e_dim=2, num_skip=5, tp=1)

        self.assertTrue(np.isnan(result.x_estimates[0]))
        self.assertTrue(np.isnan(result.y_estimates[0]))
        self.assertTrue(np.isfinite(result.correlation_x))

    def test_causalized_ccm_library_subset(self):
        t = np.linspace(0, 4 * np.pi, 260)
        x = np.sin(t)
        y = np.cos(t)
        full = causalized_ccm(x, y, tau=-1, e_dim=3, num_skip=5)
        subset = causalized_ccm(
            x,
            y,
            tau=-1,
            e_dim=3,
            num_skip=5,
            library_indices=np.arange(50),
        )

        self.assertNotEqual(full.correlation_x, subset.correlation_x)

    def test_causalized_ccm_library_subset(self):
        t = np.linspace(0, 4 * np.pi, 260)
        x = np.sin(t)
        y = np.cos(t)
        full = causalized_ccm(x, y, tau=-1, e_dim=3, num_skip=5)
        subset = causalized_ccm(
            x,
            y,
            tau=-1,
            e_dim=3,
            num_skip=5,
            library_indices=np.arange(50),
        )

        self.assertNotEqual(full.correlation_x, subset.correlation_x)

    def test_api_ccm_with_tp(self):
        t = np.linspace(0, 4 * np.pi, 260)
        x = np.sin(t)
        y = np.roll(x, 3)
        frame = pd.DataFrame({"x": x, "y": y})

        result = CCM(
            dataFrame=frame,
            columns="x",
            target="y",
            libSizes=[180, 220],
            E=2,
            tau=-1,
            Tp=1,
            num_skip=5,
            causal=True,
        )

        self.assertEqual(len(result), 2)

    def test_api_ccm_tp_validation(self):
        t = np.linspace(0, 2 * np.pi, 80)
        frame = pd.DataFrame({"x": np.sin(t), "y": np.cos(t)})

        with self.assertRaises(RuntimeError):
            CCM(
                dataFrame=frame,
                columns="x",
                target="y",
                libSizes=[20],
                E=2,
                tau=-1,
                Tp=25,
                num_skip=5,
                causal=True,
            )
    def test_ccm_conditional_argument(self):
        t = np.linspace(0, 4 * np.pi, 240)
        x = np.sin(t)
        y = np.roll(x, 2)
        z = np.cos(0.5 * t)
        frame = pd.DataFrame({"x": x, "y": y, "z": z})

        result = CCM(
            dataFrame=frame,
            columns="x",
            target="y",
            conditional="z",
            libSizes=[120, 160],
            sample=3,
            tau=-1,
            E=3,
            num_skip=5,
            causal=True,
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(list(result.columns), [
            "LibSize",
            "x:y",
            "y:x",
        ])
        self.assertTrue((result.columns[1:] == ["x:y", "y:x"]).all())

        sample_counts = result.attrs.get("SampleCount", {})
        self.assertIn(120, sample_counts)
        self.assertEqual(sample_counts[120]["x:y"], 3)
        self.assertEqual(sample_counts[120]["y:x"], 3)

        variance = result.attrs.get("Variance", {})
        self.assertIn(120, variance)
        self.assertGreaterEqual(variance[120]["x:y"], 0)
        self.assertGreaterEqual(variance[120]["y:x"], 0)

        diagnostics_mean = result.attrs.get("DiagnosticsMean", {})
        self.assertIn(120, diagnostics_mean)
        self.assertIn("x:y:var_with_cross", diagnostics_mean[120])
        self.assertIn("y:x:var_with_cross", diagnostics_mean[120])

        settings = result.attrs.get("Settings", {})
        self.assertEqual(settings.get("source"), "x")
        self.assertEqual(settings.get("target"), "y")
        self.assertEqual(settings.get("conditional"), ["z"])

    def test_conditional_libsizes_and_sampling(self):
        t = np.linspace(0, 6 * np.pi, 360)
        x = np.sin(t)
        y = 0.5 * np.roll(x, 1) + 0.5 * np.cos(0.3 * t)
        z = np.cos(0.7 * t)
        frame = pd.DataFrame({"x": x, "y": y, "z": z})

        kwargs = dict(
            dataFrame=frame,
            columns="x",
            target="y",
            conditional=["z"],
            libSizes=[200, 240],
            sample=3,
            tau=-1,
            E=3,
            num_skip=5,
            causal=True,
            seed=11,
        )

        result_one = CCM(**kwargs)
        result_two = CCM(**kwargs)

        self.assertSetEqual(set(result_one["LibSize"].unique()), {200, 240})
        sample_counts = result_one.attrs.get("SampleCount", {})
        for lib_size in (200, 240):
            self.assertEqual(sample_counts[lib_size]["x:y"], kwargs["sample"])
            self.assertEqual(sample_counts[lib_size]["y:x"], kwargs["sample"])

        variance = result_one.attrs.get("Variance", {})
        for lib_size in (200, 240):
            self.assertGreaterEqual(variance[lib_size]["x:y"], 0)
            self.assertGreaterEqual(variance[lib_size]["y:x"], 0)

        pd.testing.assert_frame_equal(result_one, result_two)

    def test_conditional_embedded_true(self):
        t = np.linspace(0, 6 * np.pi, 360)
        x = np.sin(t)
        y = 0.5 * np.roll(x, 1) + 0.5 * np.cos(0.2 * t)
        z = np.cos(0.7 * t)

        frame = pd.DataFrame({"x": x, "y": y, "z": z})

        base = CCM(
            dataFrame=frame,
            columns="x",
            target="y",
            conditional="z",
            libSizes=[200],
            E=2,
            tau=-1,
            num_skip=5,
            causal=True,
        )

        embedded_frame = pd.DataFrame(
            {
                "x0": frame["x"].iloc[1:].to_numpy(),
                "x1": frame["x"].iloc[:-1].to_numpy(),
                "y0": frame["y"].iloc[1:].to_numpy(),
                "y1": frame["y"].iloc[:-1].to_numpy(),
                "z0": frame["z"].iloc[1:].to_numpy(),
                "z1": frame["z"].iloc[:-1].to_numpy(),
            }
        )

        embedded_res = CCM(
            dataFrame=embedded_frame,
            columns=["x0", "x1"],
            target=["y0", "y1"],
            conditional=[["z0", "z1"]],
            libSizes=[200],
            E=2,
            tau=-1,
            num_skip=5,
            causal=True,
            embedded=True,
        )

        np.testing.assert_allclose(
            base.iloc[:, 1:].to_numpy(),
            embedded_res.iloc[:, 1:].to_numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        base_counts = base.attrs.get("SampleCount")
        embedded_counts = embedded_res.attrs.get("SampleCount")
        base_vals = sorted(base_counts[200].values())
        embedded_vals = sorted(embedded_counts[200].values())
        self.assertEqual(base_vals, embedded_vals)


    def test_ccm_embedded_true(self):
        t = np.linspace(0, 6 * np.pi, 260)
        x = np.sin(t)
        y = np.roll(x, 1)

        base_frame = pd.DataFrame({"x": x, "y": y})
        embedded_frame = pd.DataFrame(
            {
                "x0": x[1:],
                "x1": x[:-1],
                "y0": y[1:],
                "y1": y[:-1],
            }
        ).reset_index(drop=True)

        base_result = CCM(
            dataFrame=base_frame,
            columns="x",
            target="y",
            libSizes=[120],
            E=2,
            tau=-1,
            num_skip=5,
            causal=True,
        )

        embedded_result = CCM(
            dataFrame=embedded_frame,
            columns=["x0", "x1"],
            target=["y0", "y1"],
            libSizes=[120],
            E=2,
            tau=-1,
            num_skip=5,
            causal=True,
            embedded=True,
        )

        base_forward = base_result.iloc[:, 1].to_numpy()
        embedded_forward = embedded_result.iloc[:, 1].to_numpy()

        np.testing.assert_allclose(
            base_forward,
            embedded_forward,
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
