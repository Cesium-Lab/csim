"""COES"""

from dataclasses import dataclass

import numpy as np
from ..world import MU_EARTH_KM
from numpy.linalg import norm

####################################################################################################
#               Kepler
####################################################################################################


def kelper_eq_ellipse(M: float, e: float, tol=1e-9, max_iter=100):
    """`(M,e) -> E` \\
    Solve Kelper's equation (elliptic) `M = E - e*sin(E)` using iteration \\
    Vallado 4e Algorithm 2 p. 65

    Args:
        M (float): Mean anomaly [rad]
        e (float): Eccentricity
        tol (float, optional): Tolerance needed to break iterating. Defaults to 1e-9.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.

    Raises:
        ValueError: If the number of iterations is exceeded

    Returns:
        float: Eccentric anomaly solution [rad]
    """
    past_180_deg = M > np.pi or (M > -np.pi and M < 0)
    E_n = M - e if past_180_deg else M + e

    for _ in range(max_iter):
        E_guess = E_n + (M - E_n + e * np.sin(E_n)) / (1 - e * np.cos(E_n))

        if np.abs(E_guess - E_n) < tol:
            return E_n

        E_n = E_guess

    raise ValueError(f"Did not converge for {M=}, {e=}")


####################################################################################################
#               rv <--> coes
####################################################################################################


@dataclass
class CoesDegenExtraParams:
    """Extra angles that stand in for `raan`/`aop`/`ta` when those are undefined
    for a degenerate (circular and/or equatorial) orbit, plus the flags used to
    tell which case applies (e.g. `if extra.circular: ...`).

    Angle fields are `None` unless the orbit type makes them well-defined:

    - `arglat`: Argument of latitude [rad] -- circular inclined orbits
    - `truelon`: True longitude [rad] -- circular equatorial orbits
    - `lonper`: Longitude of periapsis [rad] -- elliptical equatorial orbits
    """

    circular: bool = False
    equatorial: bool = False
    arglat: float = None
    truelon: float = None
    lonper: float = None


def rv_to_coes(r_eci: np.ndarray, v_eci: np.ndarray, mu: float = MU_EARTH_KM, degenerate_tol: float = 1e-6):
    """Convert ECI position and velocity vectors to classical Keplerian orbital elements. \\
    Can put r and v in meters, but `mu` must be in meters as well \\
    Vallado 4e Algorithm 9 p. 113

    Args:
        r_eci (np.ndarray): Position of satellite [km] or [m] (ECI)
        v_eci (np.ndarray): Velocity of satellite [km/s] or [m/s] (ECI)
        mu (float, optional): Gravitational parameter of central body [km3/s2] or [m3/s2]. Defaults to MU_EARTH_KM [km3/s2].
        degenerate_tol (float, optional): Tolerance for degenerate orbits. Defaults to 1e-6.

    Returns:
        tuple: (`a`, `e`, `i`, `raan`, `aop`, `ta`, `extra`):
    - `a`: semi-major axis [km] or [m]
    - `e`: eccentricity
    - `i`: inclination [rad]
    - `raan`: Right ascension of ascending node [rad] (0 if equatorial -- undefined)
    - `aop`: Argument of perigee [rad] (0 if circular and/or equatorial -- undefined)
    - `ta`: True anomaly [rad] (0 if circular -- undefined)
    - `extra`: `CoesDegenExtraParams` -- `arglat`/`truelon`/`lonper`, whichever applies (see `coes_to_rv`)
    """
    # Specific relative angular momentum
    h = np.cross(r_eci, v_eci)
    h_norm = norm(h)

    # Vector pointing to ascending node
    n = np.cross([0, 0, 1], h)
    n_norm = norm(n)
    equatorial_orbit = abs(n_norm - 0.0) < degenerate_tol

    # Norms to make this easier
    r_norm = norm(r_eci)
    v_norm = norm(v_eci)
    v2 = v_norm**2
    r_dot_v = np.dot(r_eci, v_eci)

    # Eccentricity
    e = ((v2 - mu / r_norm) * r_eci - r_dot_v * v_eci) / mu
    e_norm = norm(e)
    circular_orbit = abs(e_norm - 0.0) < degenerate_tol

    # Specific Orbital Energy [km2/s2]
    E_sp = v2 / 2 - mu / r_norm

    if abs(e_norm - 1.0) > 1e-5:
        a = -mu / 2 / E_sp
    else:
        a = np.inf

    # Inclination
    i = np.arccos(h[2] / h_norm)

    # Right ascension of the ascending node [rad]
    raan = 0
    if not equatorial_orbit:
        raan = np.arccos(n[0] / n_norm)
        if n[1] < 0:
            raan = 2 * np.pi - raan

    # Argument of perigee [rad]
    aop = 0
    if not equatorial_orbit and not circular_orbit:
        aop = np.arccos(np.dot(n, e) / n_norm / e_norm)
        if e[2] < 0:
            aop = 2 * np.pi - aop

    # True anomaly [rad]
    ta = 0
    if not circular_orbit:
        ta = np.arccos(np.dot(e, r_eci) / e_norm / r_norm)
        if r_dot_v < 0:
            ta = 2 * np.pi - ta

    extra = CoesDegenExtraParams(circular=circular_orbit, equatorial=equatorial_orbit)

    # Argument of latitude [rad]
    # Circular BUT inclined
    # Just measures from the ascending node to satellite
    # no aop or ta
    if circular_orbit and not equatorial_orbit:
        arglat = np.arccos(np.dot(n, r_eci) / n_norm / r_norm)
        
        if r_eci[2] < 0: # flip if retrograde
            arglat = 2 * np.pi - arglat
        extra.arglat = arglat

    # True longitude [rad]
    # Circular AND equatorial
    # Just measure from x-axis to satellite
    # No raan, top, or ta
    if circular_orbit and equatorial_orbit:
        truelon = np.arccos(r_eci[0] / r_norm)
        
        if r_eci[1] < 0: # make positive
            truelon = 2 * np.pi - truelon
        if i > np.pi / 2: # flip if retrograde
            truelon = 2 * np.pi - truelon
        extra.truelon = truelon

    # Longitude of periapsis [rad]
    # Just equatorial and NOT circular
    # Just measure from x-axis to PERIAPSIS
    # No raan, aop
    if not circular_orbit and equatorial_orbit:
        lonper = np.arccos(e[0] / e_norm)
        if e[1] < 0: # Wrap to 2 pi if further than 180 deg around Z axis
            lonper = 2 * np.pi - lonper
        if i > np.pi / 2: # flip if retrograde
            lonper = 2 * np.pi - lonper
        extra.lonper = lonper

    return a, e_norm, i, raan, aop, ta, extra


def coes_to_rv(
    a: float,
    e: float,
    i: float,
    raan: float,
    aop: float,
    ta: float,
    mu: float = MU_EARTH_KM,
    extra: CoesDegenExtraParams = None,
):
    """
    if `mu` in units of [m3/s2] then `a`, `r`, and `v` should/will be too.

    `raan`/`aop`/`ta` are undefined for degenerate orbits (circular and/or equatorial), and are
    replaced internally by whichever field of `extra` applies -- matches Vallado's `coe2rv`
    (4e Algorithm 10 p. 118), which takes `argp`/`nu`/`arglat`/`truelon`/`lonper` all at once and
    picks the relevant angle(s) based on `e`/`i`.

    Args:
        a (float): Semi-major axis [km] or [m]
        e (float): Eccentricity
        i (float): Inclination [rad]
        raan (float): Right ascension of ascending node (RAAN) [rad]. Ignored (equatorial orbit).
        aop (float): Argument of periapsis [rad]. Ignored (circular orbit).
        ta (float): True anomaly [rad]. Ignored (circular orbit).
        mu (float, optional): Gravitational parameter of central body [km3/s2] or [m3/s2]. Defaults to MU_EARTH_KM [km3/s2]
        extra (CoesDegenExtraParams, optional): `arglat`/`truelon`/`lonper` -- required for
            degenerate orbits (see `CoesDegenExtraParams`, `rv_to_coes`).

    Raises:
        ValueError: If the orbit is degenerate and the corresponding `extra` field wasn't given

    Returns:
        tuple: (`r_eci`, `v_eci`) Position and velocity vectors (ECI) [m] and [m/s] OR [km] and [km/s]
    """

    equitorial = abs(i - 0) < 1e-5 or abs(i - np.pi) < 1e-5
    circular = abs(e - 0) < 1e-5

    if circular and equitorial:
        if extra is None or extra.truelon is None:
            raise ValueError(
                f"`extra.truelon` is required for circular equatorial orbits\n {(a, e, i, raan, aop, ta)}"
            )
        aop = 0.0
        raan = 0.0
        ta = extra.truelon
    elif circular:
        if extra is None or extra.arglat is None:
            raise ValueError(
                f"`extra.arglat` is required for circular inclined orbits\n {(a, e, i, raan, aop, ta)}"
            )
        aop = 0.0
        ta = extra.arglat
    elif equitorial:
        if extra is None or extra.lonper is None:
            raise ValueError(
                f"`extra.lonper` is required for elliptical equatorial orbits\n {(a, e, i, raan, aop, ta)}"
            )
        aop = extra.lonper
        raan = 0.0

    # Compute position and velocity in perifocal frame
    p = a * (1 - e**2)  # semi-latus rectum

    # Position in perifocal frame with components (P, Q, W) where P points to periapsis
    r_mag = p / (1 + e * np.cos(ta))
    r_pf = r_mag * np.array([np.cos(ta), np.sin(ta), 0.0])

    # Velocity in perifocal frame
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(ta), e + np.cos(ta), 0.0])

    C = calc_dcm_eci_pf(raan, i, aop)

    # Transform to ECI
    r_eci = C @ r_pf
    v_eci = C @ v_pf

    return r_eci, v_eci


def calc_dcm_eci_pf(raan: float, i: float, aop: float) -> np.ndarray:
    """Perifocal-frame DCM, expressed in ECI (perifocal -> ECI): \\
    `R = R3(-raan) * R1(-i) * R3(-aop)` And when right-multiplying, it converts from perifocal to ECI.

    Args:
        raan (float): Right ascension of ascending node (RAAN) [rad]
        i (float): Inclination [rad]
        aop (float): Argument of periapsis [rad]

    Returns:
        np.ndarray: 3x3 rotation matrix (perifocal -> ECI)
    """
    sin_raan = np.sin(raan)
    cos_raan = np.cos(raan)
    sin_i = np.sin(i)
    cos_i = np.cos(i)
    sin_aop = np.sin(aop)
    cos_aop = np.cos(aop)

    return np.array(
        [
            [
                cos_raan * cos_aop - sin_raan * sin_aop * cos_i,
                -cos_raan * sin_aop - sin_raan * cos_aop * cos_i,
                sin_raan * sin_i,
            ],
            [
                sin_raan * cos_aop + cos_raan * sin_aop * cos_i,
                -sin_raan * sin_aop + cos_raan * cos_aop * cos_i,
                -cos_raan * sin_i,
            ],
            [sin_aop * sin_i, cos_aop * sin_i, cos_i],
        ]
    )


# TODO: parabolic and hyperbolic depending on which orbits I want to simulate

# TODO: Kepler's problem?

####################################################################################################
#               Random quantities
####################################################################################################

# TODO: stuff like sma<-->period, etc.


################################################################################
#               Two-line Elements
################################################################################

# TODO: TLEs
