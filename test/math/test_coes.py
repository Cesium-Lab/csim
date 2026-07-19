import sys
import numpy as np
import pytest

sys.path.append(".")

from csim.math.coes import (
    kelper_eq_ellipse,
    rv_to_coes,
    coes_to_rv,
    calc_dcm_eci_pf,
    CoesDegenExtraParams,
)
from csim.constants import DEG_TO_RAD, RAD_TO_DEG
from csim.world import MU_EARTH, MU_EARTH_KM


def test_kelper():
    """Vallado 4e Example 2-1 p. 66"""
    M = 235.4 * DEG_TO_RAD
    e = 0.4
    E = kelper_eq_ellipse(M, e)

    assert E * RAD_TO_DEG == pytest.approx(220.512074767522)


class TestRvToCoes:
    def test_vallado(self):
        """Vallado 4e p. 115"""
        r = np.array([6524.834, 6862.875, 6448.296])  # km
        v = np.array([4.901327, 5.533756, -1.976341])  # km/s

        a, e, i, raan, aop, ta, extra = rv_to_coes(r, v)

        assert a == pytest.approx(36127.343, rel=1e-6)
        assert e == pytest.approx(0.832853, rel=1e-6)
        assert i * RAD_TO_DEG == pytest.approx(87.870, rel=1e-4)
        assert raan * RAD_TO_DEG == pytest.approx(227.898, rel=1e-5)
        assert aop * RAD_TO_DEG == pytest.approx(53.38, rel=1e-4)
        assert ta * RAD_TO_DEG == pytest.approx(92.335, rel=1e-5)
        assert not extra.circular
        assert not extra.equatorial

        """Vallado 4e p. 115 but in meters"""
        r = np.array([6524.834, 6862.875, 6448.296]) * 1000  # m
        v = np.array([4.901327, 5.533756, -1.976341]) * 1000  # m/s

        a, e, i, raan, aop, ta, extra = rv_to_coes(r, v, MU_EARTH)

        assert a == pytest.approx(36127343.0, rel=1e-6)
        assert e == pytest.approx(0.832853, rel=1e-6)
        assert i * RAD_TO_DEG == pytest.approx(87.870, rel=1e-4)
        assert raan * RAD_TO_DEG == pytest.approx(227.898, rel=1e-5)
        assert aop * RAD_TO_DEG == pytest.approx(53.38, rel=1e-4)
        assert ta * RAD_TO_DEG == pytest.approx(92.335, rel=1e-5)

    def test_random(self):
        """http://orbitsimulator.com/formulas/OrbitalElements.html"""

        r = np.array([100, 1300, -6000])  # km
        v = np.array([0.1, -1.0, 2.1])  # km/s

        a, e, i, raan, aop, ta, extra = rv_to_coes(r, v)

        assert a == pytest.approx(3203.754267903962, rel=1e-6)
        assert e == pytest.approx(0.9955257908191099, rel=1e-6)
        assert i * RAD_TO_DEG == pytest.approx(93.90569432920545, rel=1e-4)
        assert raan * RAD_TO_DEG == pytest.approx(283.91249467651966, rel=1e-5)
        assert aop * RAD_TO_DEG == pytest.approx(77.26657593515651, rel=1e-4)
        assert ta * RAD_TO_DEG == pytest.approx(181.10299292301673, rel=1e-5)

    def test_circular(self):
        """http://orbitsimulator.com/formulas/OrbitalElements.html"""

        r = np.array([6000, 4713.675910949817, 1000])  # km
        v = np.array([4.446, -5.658, 0])  # km/s (circular orbit)

        a, e, i, raan, _, _, extra = rv_to_coes(r, v, degenerate_tol=5e-4) # Online calculator uses tolerance of 1e-3

        assert a == pytest.approx(7692.648, rel=1e-4)
        assert e == pytest.approx(0, abs=1e-3) 
        assert i * RAD_TO_DEG == pytest.approx(172.53, rel=1e-4)
        assert raan * RAD_TO_DEG == pytest.approx(128.1599, rel=1e-5)
        # argument of perigee and true anomaly are undefined for circular orbits (can be any value)
        assert extra.circular
        assert not extra.equatorial
        assert extra.arglat is not None

    def test_0_inclination(self):
        """http://orbitsimulator.com/formulas/OrbitalElements.html"""

        r = np.array([6800, 0, 0])  # km
        v = np.array([0, 8.656278815328648, 0])  # km/s

        a, e, i, _, _, _, extra = rv_to_coes(r, v)

        assert a == pytest.approx(9421.974578532005, rel=1e-4)
        assert e == pytest.approx(0.2782829179465395, abs=1e-4)
        assert i * RAD_TO_DEG == pytest.approx(0, rel=1e-4)
        # raan, aop and ta are undefined for circular orbits (can be any value)
        assert extra.equatorial
        assert not extra.circular
        assert extra.lonper is not None

    def test_90_inclination(self):
        """http://orbitsimulator.com/formulas/OrbitalElements.html"""

        r = np.array([6686, 0, 968.35])  # km
        v = np.array([0, 0, -5.899127972862638])  # km/s

        a, e, i, raan, aop, ta, extra = rv_to_coes(r, v)

        assert a == pytest.approx(4790.6431, rel=1e-5)
        assert e == pytest.approx(0.4305248975978197, rel=1e-4)
        assert i * RAD_TO_DEG == pytest.approx(90, rel=1e-5)
        assert raan * RAD_TO_DEG == pytest.approx(
            180, rel=1e-5
        )  # Since going down initially, ascends on other side
        assert aop * RAD_TO_DEG == pytest.approx(340.552, rel=1e-5)
        assert ta * RAD_TO_DEG == pytest.approx(191.206, rel=1e-5)

    def test_hyperbolic_a(self):
        """hyperbolic orbit semi-major axis should be negative"""
        r = np.array([7000.0, 0.0, 0.0])  # km
        v = np.array([0.0, 15.0, 0.0])  # km/s

        a, e, i, raan, aop, ta, extra = rv_to_coes(r, v)

        assert e > 1  # hyperbolic orbit
        assert a < 0
        assert a == pytest.approx(-3587.3055571396117, rel=1e-6)

    def test_180_inclination(self):
        """http://orbitsimulator.com/formulas/OrbitalElements.html"""

        r = np.array([7000, 2, 0])  # km
        v = np.array([1.2, -2, 0])  # km/s

        a, e, i, _, _, _, extra = rv_to_coes(r, v)

        assert a == pytest.approx(3675.56885, rel=1e-5)
        assert e == pytest.approx(0.930685, rel=1e-4)
        assert i * RAD_TO_DEG == pytest.approx(180, rel=1e-5)
        # raan, aop, and ta are undefined for equitorial orbits (can be any value)
        assert extra.equatorial
        assert extra.lonper is not None

    def test_degenerate_tol(self):
        """A near-circular orbit (e=1e-5) should flip between circular/not
        depending on `degenerate_tol`, since that's exactly the threshold
        `circular_orbit = abs(e_norm) < degenerate_tol` is checking."""
        a = 7000.0  # km
        e = 1e-5
        i = 45 * DEG_TO_RAD
        raan = 30 * DEG_TO_RAD
        aop = 10 * DEG_TO_RAD
        ta = 20 * DEG_TO_RAD

        r, v = coes_to_rv(a, e, i, raan, aop, ta)

        # default tol (1e-6): e_norm=1e-5 is not < 1e-6 -> not circular
        _, _, _, _, _, _, extra_default = rv_to_coes(r, v)
        assert not extra_default.circular
        assert extra_default.arglat is None

        # looser tol (1e-4): e_norm=1e-5 is < 1e-4 -> circular
        _, _, _, _, _, _, extra_loose = rv_to_coes(r, v, degenerate_tol=1e-4)
        assert extra_loose.circular
        assert extra_loose.arglat is not None


class TestCoesToRv:
    def test_vallado(self):
        """Vallado 4e Example 2-6 p. 119"""

        # p given as 11067.79 but p = a(1-e**2)
        a = 36126.64  # km
        e = 0.83285
        i = 87.87 * DEG_TO_RAD
        raan = 227.89 * DEG_TO_RAD
        aop = 53.38 * DEG_TO_RAD
        ta = 92.335 * DEG_TO_RAD

        r, v = coes_to_rv(a, e, i, raan, aop, ta)

        assert np.allclose(r, [6525.344, 6861.535, 6449.125], atol=0.1)
        assert np.allclose(v, [4.902276, 5.533124, -1.975709], atol=0.0001)

        # Convert to m now
        a = a * 1000
        r, v = coes_to_rv(a, e, i, raan, aop, ta, MU_EARTH)  # use m3/s2 constant
        assert np.allclose(r, [6525344, 6861535, 6449125], atol=0.1)
        assert np.allclose(v, [4902.276, 5533.124, -1975.709], atol=0.0001)

    def test_random(self):
        """http://orbitsimulator.com/formulas/OrbitalElements.html"""

        a = 3203.754267903962  # km
        e = 0.9955257908191099
        i = 93.90569432920545 * DEG_TO_RAD
        raan = 283.91249467651966 * DEG_TO_RAD
        aop = 77.26657593515651 * DEG_TO_RAD
        ta = 181.10299292301673 * DEG_TO_RAD

        r, v = coes_to_rv(a, e, i, raan, aop, ta)

        assert np.allclose(r, [100.0, 1300.0, -6000.0], atol=0.00001)
        assert np.allclose(v, [0.1, -1.0, 2.1], atol=0.0001)

    def test_90_inclination(self):
        """http://orbitsimulator.com/formulas/OrbitalElements.html"""

        a = 4848.9197268518665  # km
        e = 0.3938736352565609
        i = 90.0 * DEG_TO_RAD
        raan = 180.0 * DEG_TO_RAD
        aop = 353.8847334059262 * DEG_TO_RAD
        ta = 177.87427980631205 * DEG_TO_RAD

        r, v = coes_to_rv(a, e, i, raan, aop, ta)

        assert np.allclose(r, [6686, 0.0, 968.35], atol=1e-6)
        assert np.allclose(v, [1.0, 0.0, -5.899127972862638], atol=1e-4)


class TestCoesToRvDegenerateCases:
    """Tests from Vallado's Ex2_5.m since new functionality for degenerate orbits.

    Round-trips rv_to_coes -> coes_to_rv for the degenerate orbit types (circular
    inclined, elliptical equatorial, circular equatorial) that coes_to_rv used to
    raise NotImplementedError for. r/v test vectors from Vallado's Ex2_5.m
    (stressing cases for coe/rv conversions), which pairs rv2coe with coe2rv the
    same way these tests do."""

    def _assert_round_trip(self, r, v):
        a, e, i, raan, aop, ta, extra = rv_to_coes(r, v)

        r_out, v_out = coes_to_rv(a, e, i, raan, aop, ta, extra=extra)

        assert np.allclose(r_out, r, rtol=1e-4)
        assert np.allclose(v_out, v, rtol=1e-4)

    def test_circular_inclined_u45(self):
        r = np.array([-2693.34555010128, 6428.43425355863, 4491.37782050409])
        v = np.array([-3.95484712246016, -4.28096585381370, 3.75567104538731])
        self._assert_round_trip(r, v)

    def test_circular_inclined_u315(self):
        r = np.array([-7079.68834483379, 3167.87718823353, -2931.53867301568])
        v = np.array([1.77608080328182, 6.23770933190509, 2.45134017949138])
        self._assert_round_trip(r, v)

    def test_elliptical_equatorial_w20(self):
        r = np.array([-22739.1086596208, -22739.1086596208, 0.0])
        v = np.array([2.48514004188565, -2.02004112073465, 0.0])
        self._assert_round_trip(r, v)

    def test_elliptical_equatorial_w240(self):
        r = np.array([28242.3662822040, 2470.8868808397, 0.0])
        v = np.array([0.66575215057746, -3.62533022188304, 0.0])
        self._assert_round_trip(r, v)

    def test_circular_equatorial_l65(self):
        r = np.array([6199.6905946008, 13295.2793851394, 0.0])
        v = np.array([-4.72425923942564, 2.20295826245369, 0.0])
        self._assert_round_trip(r, v)

    def test_circular_equatorial_l65_retrograde(self):
        """Same case, but i=180 (retrograde equatorial)."""
        r = np.array([6199.6905946008, -13295.2793851394, 0.0])
        v = np.array([-4.72425923942564, -2.20295826245369, 0.0])
        self._assert_round_trip(r, v)

    def test_missing_arglat_raises(self):
        # In the case of circular inclined
        with pytest.raises(ValueError):
            coes_to_rv(a=7000, e=0.0, i=45 * DEG_TO_RAD, raan=0, aop=0, ta=0)

    def test_missing_truelon_raises(self):
        # In the case of circular equatorial
        with pytest.raises(ValueError):
            coes_to_rv(a=7000, e=0.0, i=0.0, raan=0, aop=0, ta=0)

    def test_missing_lonper_raises(self):
        # In the case of elliptical equatorial
        with pytest.raises(ValueError):
            coes_to_rv(a=7000, e=0.1, i=0.0, raan=0, aop=0, ta=0)

    def test_extra_with_wrong_field_raises(self):
        # extra given, but not the field this degenerate case needs
        with pytest.raises(ValueError):
            coes_to_rv(
                a=7000,
                e=0.0,
                i=0.0,
                raan=0,
                aop=0,
                ta=0,
                extra=CoesDegenExtraParams(arglat=1.0),
            )


class TestDcmEciPf:
    def test_vallado(self):
        """Perifocal -> ECI rotation matrix, using the perifocal-frame r/v and
        the expected ECI r/v from Vallado 4e Example 2-6"""
        a = 36126.64  # km
        e = 0.83285
        i = 87.87 * DEG_TO_RAD
        raan = 227.89 * DEG_TO_RAD
        aop = 53.38 * DEG_TO_RAD
        ta = 92.335 * DEG_TO_RAD

        p = a * (1 - e**2)
        r_mag = p / (1 + e * np.cos(ta))
        r_pf = r_mag * np.array([np.cos(ta), np.sin(ta), 0.0])
        v_pf = np.sqrt(MU_EARTH_KM / p) * np.array([-np.sin(ta), e + np.cos(ta), 0.0])

        C = calc_dcm_eci_pf(raan, i, aop)

        # Rotation matrix
        assert np.linalg.det(C) == pytest.approx(1, abs=1e-9)

        # Transforms vectors correctly (same expected values as TestCoesToRv.test_vallado)
        assert np.allclose(C @ r_pf, [6525.344, 6861.535, 6449.125], atol=0.1)
        assert np.allclose(C @ v_pf, [4.902276, 5.533124, -1.975709], atol=0.0001)
