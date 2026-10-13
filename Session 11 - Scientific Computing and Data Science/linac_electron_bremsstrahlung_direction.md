# Bremsstrahlung from a LINAC Electron Striking a Target

Companion notes for `linac_electron_bremsstrahlung_direction.py`.

This document explains the physics behind the script and then walks through
the code one piece at a time. Open it in VS Code with **Markdown: Open
Preview** (`Ctrl+Shift+V`) so the equations render.

The two earlier documents in this folder cover an electron *inside* the
LINAC. `linac_electron_radiation_direction.md` shows where a smoothly
accelerated electron radiates, and `linac_electron_radiation_power.md` shows
how little power that costs. This document picks up at the far end of the
machine, where the beam slams into a target and the real X-rays are made.

---

## 1. The Question

A LINAC delivers an electron beam to a metal target. The electrons stop in
the target, and X-rays come out. This is **bremsstrahlung**, German for
"braking radiation". Two questions follow:

- In which directions do the X-rays leave?
- How does that pattern change with the energy of the beam?

The short answers:

- **At low energy the X-rays go sideways.** A 10 keV electron radiates in a
  doughnut around the beam, brightest at almost 90 degrees from it and dark
  straight ahead and straight behind.
- **At high energy they go forward in a narrow cone.** A 10 MeV electron
  radiates into a cone less than 3 degrees wide. At 100 MeV the cone is a
  needle a third of a degree wide.
- **The cone half-angle is $1/\gamma$,** where $\gamma$ is the Lorentz
  factor. For an electron $1/\gamma = m_e c^2 / E$, so the cone shrinks in
  proportion to the energy.

The script draws the two-dimensional radiation pattern at five energies side
by side, then plots the cone angle against energy from 1 keV to 1 GeV.

---

## 2. What Bremsstrahlung Is

A fast electron entering a target passes close to atomic nuclei. The Coulomb
field of a nucleus is enormous at close range and it jerks the electron's
velocity, bending its path and slowing it down. Any change of velocity means
acceleration, and an accelerated charge radiates. Because the nucleus is
thousands of times heavier than the electron, the electron takes essentially
all of the jolt and the nucleus hardly moves.

This is the mechanism behind:

- the continuous X-ray spectrum of a dental or medical X-ray tube, where the
  electrons carry tens of keV,
- the treatment beam of a medical LINAC, where 6 MeV to 25 MeV electrons hit
  a tungsten target,
- the photon beams of research accelerators at hundreds of MeV and beyond.

Two things the model in this document leaves out:

- **Characteristic X-rays.** A fast electron can also knock an inner-shell
  electron out of a target atom. When the hole is refilled, the atom emits a
  photon at a fixed energy. Those sharp lines sit on top of the smooth
  bremsstrahlung continuum and come from a different process.
- **The photon energy spectrum.** Bremsstrahlung photons carry anything from
  zero up to the full kinetic energy of the electron. This document is about
  *direction*, and the classical model used here has the same angular
  pattern at every photon energy, so the spectrum does not enter.

---

## 3. Radiation From a Sudden Change of Velocity

### 3.1 The radiation field

The field of a moving point charge is the **Liénard–Wiechert field**. The
part that carries energy to infinity is the acceleration field,

$$
\vec{E}_{\text{rad}} = \frac{q}{4\pi\varepsilon_0 c}\,
\frac{\hat{n}\times\big[(\hat{n}-\vec{\beta})\times\dot{\vec{\beta}}\big]}
{(1-\hat{n}\cdot\vec{\beta})^3\,R}
$$

evaluated at the retarded time. Here $\vec{\beta} = \vec{v}/c$,
$\dot{\vec{\beta}} = \vec{a}/c$, and $\hat{n}$ is the unit vector from the
charge toward the observer at distance $R$.
`linac_electron_radiation_direction.md` starts from this same expression.

### 3.2 Energy per unit frequency per unit solid angle

A collision is a brief event, not a steady state, so the natural question is
not "how much power is radiated right now" but "how much energy reaches the
observer altogether, and at what frequencies". Take the Fourier transform of
the radiation field, apply Parseval's theorem, and the energy radiated per
unit frequency per unit solid angle comes out as

$$
\frac{d^2 I}{d\omega\, d\Omega}
= \frac{q^2 \omega^2}{16\pi^3\varepsilon_0 c}
\left|\int_{-\infty}^{\infty}
\hat{n}\times(\hat{n}\times\vec{\beta})\,
e^{\,i\omega\,(t - \hat{n}\cdot\vec{r}(t)/c)}\,dt\right|^2 .
$$

The integrand contains only $\vec{\beta}$, not $\dot{\vec{\beta}}$. That
simplification comes from the identity

$$
\frac{\hat{n}\times\big[(\hat{n}-\vec{\beta})\times\dot{\vec{\beta}}\big]}
{(1-\hat{n}\cdot\vec{\beta})^2}
= \frac{d}{dt}\left[\frac{\hat{n}\times(\hat{n}\times\vec{\beta})}
{1-\hat{n}\cdot\vec{\beta}}\right]
$$

which lets the time derivative be moved onto the exponential by integration
by parts. The exponential's derivative supplies the factor
$\omega\,(1-\hat{n}\cdot\vec{\beta})$, and the net effect is the form above.

### 3.3 A collision that is over in an instant

Model the collision as a **velocity that jumps** from $\vec{\beta}_{\text{in}}$
to $\vec{\beta}_{\text{out}}$ at $t = 0$, at the origin. The integral then
splits into two pieces. Before the jump $\vec{r}(t) = \vec{\beta}_{\text{in}}
c\,t$ and the phase is $\omega t\,(1 - \hat{n}\cdot\vec{\beta}_{\text{in}})$.
Integrating from $-\infty$ to $0$ gives

$$
\int_{-\infty}^{0} e^{\,i\omega t(1-\hat{n}\cdot\vec{\beta}_{\text{in}})}\,dt
= \frac{1}{i\omega\,(1-\hat{n}\cdot\vec{\beta}_{\text{in}})} .
$$

The piece after the jump gives the same with $\vec{\beta}_{\text{out}}$ and
the opposite sign. The factor $1/\omega$ cancels the $\omega^2$ in front, and
the result is

$$
\boxed{\;
\frac{d^2 I}{d\omega\, d\Omega}
= \frac{q^2}{16\pi^3\varepsilon_0 c}
\left|\vec{A}(\vec{\beta}_{\text{out}}) - \vec{A}(\vec{\beta}_{\text{in}})\right|^2,
\qquad
\vec{A}(\vec{\beta}) = \frac{\hat{n}\times(\hat{n}\times\vec{\beta})}
{1-\hat{n}\cdot\vec{\beta}}
\;}
$$

This is the formula the script evaluates. Three things to notice:

- **The frequency has dropped out.** The spectrum is flat. A sudden jump
  contains every frequency equally, in the same way that a sharp click
  contains every pitch. In reality the collision takes a short but finite
  time $\tau$, and the flat spectrum holds only for $\omega \ll 1/\tau$. The
  angular pattern is the same at every frequency in that range, which is why
  the script can plot a single pattern per energy.
- **Only the velocities before and after matter.** The details of how the
  velocity changed during the collision do not enter.
- **$\vec{A}$ is linear in $\vec{\beta}$,** so the two terms can interfere.
  If the velocity barely changes, the two terms nearly cancel and almost
  nothing is radiated, which is as it should be.

### 3.4 Stopping dead

The simplest collision is one that stops the electron completely, so
$\vec{\beta}_{\text{out}} = 0$ and only one term survives. With
$\vec{\beta}_{\text{in}}$ along the beam and $\theta$ the angle between the
beam and $\hat{n}$, the vector $\hat{n}\times(\hat{n}\times\vec{\beta})$ is
the part of $\vec{\beta}$ perpendicular to the line of sight, with magnitude
$\beta\sin\theta$. Then

$$
\boxed{\;
\frac{d^2 I}{d\omega\, d\Omega}
= \frac{q^2}{16\pi^3\varepsilon_0 c}\,
\frac{\beta^2\sin^2\theta}{(1-\beta\cos\theta)^2}
\;}
$$

The $\sin^2\theta$ in the numerator says there is **no radiation exactly
along the beam**, forward or backward, at any energy. The denominator says the
pattern is **thrown forward** as $\beta \to 1$, because $1 - \beta\cos\theta$
is smallest near $\theta = 0$.

### 3.5 Comparing with the smooth-acceleration pattern

`linac_electron_radiation_direction.py` plots the power pattern of an electron
whose acceleration is parallel to its velocity, the Liénard result. Both
patterns describe straight-line braking, so it is worth putting them side by
side.

| | Smooth acceleration (Liénard) | Sudden stop (this script) |
| --- | --- | --- |
| Quantity | Power per solid angle, $dP/d\Omega$ | Energy per frequency per solid angle, $d^2I/d\omega\,d\Omega$ |
| Pattern | $\dfrac{\sin^2\theta}{(1-\beta\cos\theta)^5}$ | $\dfrac{\sin^2\theta}{(1-\beta\cos\theta)^2}$ |
| Peak angle | $\cos\theta_{\max} = \dfrac{\sqrt{1+15\beta^2}-1}{3\beta}$ | $\cos\theta_{\max} = \beta$ |
| High-energy limit | $\theta_{\max} \approx \dfrac{1}{2\gamma}$ | $\theta_{\max} \approx \dfrac{1}{\gamma}$ |

Both share the $\sin^2\theta$ numerator, so both are hollow cones with a dark
axis. The exponents differ because the two quantities are built differently.
Instantaneous power squares the field, which carries
$(1-\beta\cos\theta)^{-3}$, and then converts from observer time to emitter
time, which costs one more power. The sudden-stop energy comes from a time
integral in which each velocity contributes a single factor of
$(1-\beta\cos\theta)^{-1}$ to the amplitude, and squaring gives the exponent
2. Both say the same thing physically: the faster the electron, the more
sharply forward the radiation goes, with a characteristic angle of order
$1/\gamma$.

---

## 4. The Angle of Peak Emission

Write the angular part of the sudden-stop pattern as

$$
f(\theta) = \frac{\sin^2\theta}{(1-\beta\cos\theta)^2} .
$$

Differentiate with the quotient rule:

$$
\frac{df}{d\theta}
= \frac{2\sin\theta\cos\theta\,(1-\beta\cos\theta) - 2\beta\sin^3\theta}
{(1-\beta\cos\theta)^3}
= \frac{2\sin\theta\,\big[\cos\theta - \beta\cos^2\theta - \beta\sin^2\theta\big]}
{(1-\beta\cos\theta)^3}
= \frac{2\sin\theta\,(\cos\theta - \beta)}{(1-\beta\cos\theta)^3} .
$$

The factor $\sin\theta$ vanishes on the axis, where the pattern is zero, so
the peak is where the bracket vanishes:

$$
\boxed{\;\cos\theta_{\max} = \beta\;}
$$

This is as clean a result as the subject offers. Two equivalent forms are
just as useful:

$$
\sin\theta_{\max} = \sqrt{1-\beta^2} = \frac{1}{\gamma},
\qquad
f(\theta_{\max}) = \frac{1-\beta^2}{(1-\beta^2)^2} = \gamma^2 .
$$

So the peak lies exactly where $\sin\theta = 1/\gamma$, and the peak
brightness grows as $\gamma^2$. The script prints $\theta_{\max}$ found by a
numerical search alongside $\arccos\beta$, and the two agree to every digit
shown.

### 4.1 Checking the limits

- **Slow electron,** $\beta \to 0$: $\cos\theta_{\max} \to 0$, so
  $\theta_{\max} \to 90^\circ$. The pattern is the $\sin^2\theta$ doughnut of
  a dipole antenna, brightest sideways.
- **Fast electron,** $\beta \to 1$: $\sin\theta_{\max} = 1/\gamma$ is small,
  so $\theta_{\max} \approx 1/\gamma$. For an electron,
  $1/\gamma = m_e c^2/E = 0.511\ \text{MeV}/E$, so the peak angle in radians
  is the rest energy divided by the total energy.

### 4.2 The numbers the script prints

| Kinetic energy | $\beta$ | $\gamma$ | $\theta_{\max}$ | $1/\gamma$ in degrees |
| --- | --- | --- | --- | --- |
| 10 keV | 0.195 | 1.020 | 78.76° | 56.20° |
| 100 keV | 0.548 | 1.196 | 56.75° | 47.92° |
| 1 MeV | 0.941 | 2.957 | 19.77° | 19.38° |
| 10 MeV | 0.9988 | 20.57 | 2.787° | 2.785° |
| 100 MeV | 0.99999 | 196.7 | 0.291° | 0.291° |

The $1/\gamma$ column is only meaningful once the angle is small. By 1 MeV
it is within 2 percent of the exact value, and by 10 MeV the two agree to
three figures.

---

## 5. How Much of the Energy Is in the Cone

$\theta_{\max}$ says where the pattern is brightest. It does not say where the
energy goes, because solid angle grows with $\theta$ and a dim but wide region
can hold more energy than a bright but narrow one. The script integrates the
pattern over the sphere and reports two more numbers: the share of the energy
emitted into the forward hemisphere, and the half-angle of the cone that holds
half the energy.

### 5.1 The energy is spread logarithmically

Near the axis of a fast electron, with $\theta \ll 1$,

$$
1-\beta\cos\theta \approx \frac{1+\gamma^2\theta^2}{2\gamma^2},
\qquad\text{so}\qquad
f(\theta) \approx \frac{4\gamma^4\theta^2}{(1+\gamma^2\theta^2)^2} .
$$

The energy per unit $\theta$ is $f(\theta)\sin\theta \approx f(\theta)\,\theta$,
which is

$$
\frac{4\gamma^4\theta^3}{(1+\gamma^2\theta^2)^2}
\;\longrightarrow\;
\frac{4}{\theta}
\qquad\text{for }\theta \gg \frac{1}{\gamma} .
$$

A $1/\theta$ distribution puts the **same energy into every factor-of-two
range of angle**. The band from $1/\gamma$ to $2/\gamma$ carries as much as
the band from $2/\gamma$ to $4/\gamma$, and so on out to angles of order one
radian where the small-angle approximation fails. The peak at $1/\gamma$ is
bright, but most of the energy is in the long tail beyond it.

Integrating the small-angle form from $0$ to $\theta$ gives the energy inside
a cone of half-angle $\theta$:

$$
F(\theta) = 2\left[\ln(1+\gamma^2\theta^2) + \frac{1}{1+\gamma^2\theta^2} - 1\right] .
$$

The exact total over the whole sphere, from the unapproximated pattern, is

$$
\int \frac{\beta^2\sin^2\theta}{(1-\beta\cos\theta)^2}\,d\Omega
= 4\pi\left[\frac{1}{\beta}\ln\frac{1+\beta}{1-\beta} - 2\right]
\;\approx\; 8\pi\,\big[\ln(2\gamma) - 1\big] \quad (\gamma \gg 1).
$$

Setting $F(\theta_{1/2})$ equal to half of that, and using
$\gamma\theta \gg 1$ to drop the small terms, gives
$\ln(1+\gamma^2\theta_{1/2}^2) \approx \ln(2\gamma)$, so

$$
\boxed{\;\theta_{1/2} \approx \sqrt{\frac{2}{\gamma}}\;}
$$

At 100 MeV this predicts $\sqrt{2/196.7} = 0.1008$ rad $= 5.78^\circ$, and the
script's numerical integration reports $5.77^\circ$. The **half-energy cone
shrinks only as $1/\sqrt{\gamma}$**, far more slowly than the peak angle, which
shrinks as $1/\gamma$. Both numbers describe the same pattern, and the plot's
sixth panel shows them pulling apart as the energy rises.

### 5.2 The numbers the script prints

| Kinetic energy | Forward share | $\theta_{\max}$ | Half-energy half-angle |
| --- | --- | --- | --- |
| 10 keV | 0.573 | 78.8° | 84.4° |
| 100 keV | 0.710 | 56.8° | 72.6° |
| 1 MeV | 0.898 | 19.8° | 45.3° |
| 10 MeV | 0.965 | 2.79° | 17.7° |
| 100 MeV | 0.981 | 0.291° | 5.77° |

The forward share creeps toward 1 but never gets there quickly. For
$\beta \to 1$ the backward hemisphere holds a fixed amount,
$2\pi(2\ln 2 - 1)$ in the units of the integral above, while the forward
hemisphere grows as $\ln\gamma$. At 100 MeV that leaves 2 percent of the
energy going backward even though the peak is a needle pointing forward.

---

## 6. Deflection Instead of Stopping

A real bremsstrahlung event rarely stops the electron. The nucleus deflects
it, usually through a small angle, and the electron carries on with most of
its energy. The script handles this through two settings in `main()`:
`exit_speed_fraction` sets the speed after the collision as a fraction of the
speed before, and `exit_angle_deg` sets the direction after the collision,
measured from the beam toward $+y$. The defaults, 0 and 0, give the dead stop
of Section 3.4.

With a nonzero exit velocity both terms of the boxed formula in Section 3.3
survive. $\vec{A}(\vec{\beta}_{\text{in}})$ is a hollow cone of half-angle
$1/\gamma$ around the incoming direction, $\vec{A}(\vec{\beta}_{\text{out}})$
is the same cone around the outgoing direction, and the two interfere. What
the pattern looks like depends on how the deflection angle $\delta$ compares
with $1/\gamma$.

- **Large deflection,** $\delta \gg 1/\gamma$. The two cones are well
  separated and barely interfere. Set `exit_speed_fraction = 1.0` and
  `exit_angle_deg = 30.0` at 10 MeV: the map shows one cone around the beam
  and a second around the exit path. In the plane each cone appears as a pair
  of peaks straddling its axis, at about 3 degrees either side of the beam
  and at 27 and 33 degrees around the exit path.
- **Small deflection,** $\delta \ll 1/\gamma$. The two terms nearly cancel.
  The difference is proportional to $\delta$, so the radiated energy falls as
  $\delta^2$. The shape changes too. The pattern is no longer hollow but is
  brightest on the axis, halfway between the two paths, with nulls at
  $\pm 1/\gamma$ and weak side lobes beyond them. This is the pattern of
  acceleration *perpendicular* to the velocity, which is what a small turn at
  constant speed is. Try `exit_angle_deg = 0.2` at 10 MeV, where $1/\gamma$
  is 2.8 degrees. Each map is normalized to its own peak, so the maps show
  the change of shape but not the drop in brightness.

A small deflection radiates little, which is why a thin target converts only
a small fraction of the beam energy into photons.

What the model does and does not capture about a real target:

- **Thin target.** Each electron suffers one or a few collisions and leaves.
  The photon cone follows the electron direction with the $1/\gamma$ width
  derived here. The full quantum calculation (Bethe and Heitler) changes the
  spectrum but keeps the characteristic angle $m_e c^2/E$.
- **Thick target.** The electrons scatter many times before they stop, so
  their own directions spread out. The emerging photon cone is the $1/\gamma$
  cone smeared by that spread, and it is wider than this model predicts.
- **The flattening filter.** A medical LINAC treatment beam is strongly
  forward-peaked for exactly the reason shown in the 10 MeV panel. The beam
  head carries a cone-shaped metal filter, thickest on the axis, that absorbs
  more of the center than the edges to produce a flat dose profile across
  the field.

---

## 7. Walking Through the Code

### 7.1 Header and imports

```python
#!/usr/bin/env -S uv run
"""linac_electron_bremsstrahlung_direction.py"""
```

The script uses `scipy.constants` for the electron mass, charge, and the
speed of light; `scipy.integrate` for `trapezoid` and `cumulative_trapezoid`,
which do the sphere integral in Section 5; and `scipy.optimize` for
`minimize_scalar`, which refines the peak angle. `LogNorm` gives the maps a
logarithmic color scale and `Rectangle` draws the target foil.

The module constant `REST_ENERGY_MEV` is $m_e c^2$ converted to MeV, about
0.511. Every conversion from kinetic energy to speed goes through it.

### 7.2 `speed_from_kinetic_energy(kinetic_energy_mev)`

Returns $(\beta, \gamma)$ for a kinetic energy $T$. The total energy is
$\gamma m_e c^2 = T + m_e c^2$, so $\gamma = 1 + T/(m_e c^2)$ and then
$\beta = \sqrt{1 - 1/\gamma^2}$. Computing $\gamma$ first and $\beta$ from it
keeps full precision at high energy, where $\beta$ differs from 1 by only a
few parts in $10^8$.

### 7.3 `velocity_term(n, beta_vec)`

Evaluates $\vec{A}(\vec{\beta})$ from Section 3.3 for a whole array of
observation directions at once. `n` has shape `(..., 3)`, one unit vector per
observation point, and `beta_vec` is a single 3-vector.

```python
n_dot_beta = n @ beta_vec
transverse = n * n_dot_beta[..., np.newaxis] - beta_vec
return transverse / (1 - n_dot_beta)[..., np.newaxis]
```

The triple product is expanded with the identity
$\hat{n}\times(\hat{n}\times\vec{\beta}) = \hat{n}(\hat{n}\cdot\vec{\beta}) - \vec{\beta}$,
which avoids cross products and works for any array shape. The `np.newaxis`
slices broadcast the scalar dot product across the three vector components.

### 7.4 `radiation_pattern(n, beta_in, beta_out)`

Subtracts the two velocity terms and sums the squares of the components to
get $|\vec{A}(\vec{\beta}_{\text{out}}) - \vec{A}(\vec{\beta}_{\text{in}})|^2$.
Every other function builds on this one. The overall constant
$q^2/(16\pi^3\varepsilon_0 c)$ is left out because every plot is normalized.

### 7.5 `in_plane_pattern(theta, beta_in, beta_out)`

Builds observation vectors $(\cos\theta, \sin\theta, 0)$ in the plane that
holds the beam and the exit path, and hands them to `radiation_pattern`. This
is the function the maps use, since each map is a slice through that plane.
The sign of $\theta$ matters when the electron is deflected toward $+y$,
because the pattern is then not symmetric about the beam.

### 7.6 `peak_angle(beta_in, beta_out)`

Finds $\theta_{\max}$ numerically rather than from the formula, so it also
works for a deflected electron where no closed form exists. A scan of 7201
angles between 0 and $\pi$, a step of 0.025 degrees, locates the brightest
lobe. That is too coarse on its own at 1 GeV, where the lobe is 0.03 degrees
wide, so `minimize_scalar` with the `bounded` method then searches between the
neighboring scan points and converges to the peak within $10^{-10}$ radians.
The search is restricted to $0 \le \theta \le \pi$; the lobes of a deflected
electron lie on the $+y$ side or straddle the beam, so nothing is missed.

### 7.7 `energy_cone(beta_in, beta_out, fraction=0.5)`

Integrates the pattern over the full sphere to produce the two numbers in
Section 5. It lays down a grid of 6001 polar angles by 181 azimuths, builds
the observation vectors

$$
\hat{n} = (\cos\theta,\ \sin\theta\cos\phi,\ \sin\theta\sin\phi),
$$

and evaluates the pattern at every point. `trapezoid` over $\phi$ and a
factor of $\sin\theta$ turn that into energy per unit $\theta$, and
`cumulative_trapezoid` turns it into the running total $F(\theta)$. Two
interpolations read off the answers: the value of the running total at
$\theta = \pi/2$ divided by the grand total is the forward share, and the
angle at which the running total reaches `fraction` of the grand total is the
cone half-angle. The grid step of 0.03 degrees puts about ten points across
the 100 MeV peak, enough for three-figure answers.

### 7.8 `energy_label(kinetic_energy_mev)`

Formats an energy as keV, MeV, or GeV so that titles and the table read
naturally.

### 7.9 `main()`: the settings

```python
# TODO: Change the electron kinetic energies to compare (MeV)
kinetic_energies_mev = [0.01, 0.1, 1.0, 10.0, 100.0]

# TODO: Change what the collision does to the electron. ...
exit_speed_fraction = 0.0
exit_angle_deg = 0.0
```

The five energies span an X-ray tube (10 keV), a medical LINAC (10 MeV), and
a research machine (100 MeV). Five is the number of map panels in the figure.

### 7.10 `main()`: the table

A loop over the energies computes $\beta$ and $\gamma$, builds the two
velocity vectors, and calls `peak_angle` and `energy_cone`. Each row of the
table shows the energy, $\beta$, $\gamma$, the numerical $\theta_{\max}$, the
analytic $\arccos\beta$, the $1/\gamma$ estimate in degrees, the forward
share, and the half-energy half-angle. The tuple appended to `results` is
reused by the plots so nothing is computed twice.

### 7.11 `main()`: the maps

The figure is a 2 by 3 grid with constrained layout. The first five axes hold
the maps and the sixth holds the cone-angle curve.

```python
x, y = np.meshgrid(np.linspace(-extent, extent, n_points), ...)
r = np.hypot(x, y)
theta = np.arctan2(y, x)
```

A 601 by 601 grid of points fills the plane. Each point sees the collision at
the origin under the angle `theta`, taken with its sign for the reason given
in Section 7.5. The far-field intensity is the pattern divided by $r^2$, with
the region inside `r_min` masked out because $1/r^2$ diverges there.

```python
norm = LogNorm(vmin=1e-6 / r_min**2, vmax=1 / r_min**2)
```

One `LogNorm` is shared by all five maps so that their colors can be compared
directly. Each map is normalized to its own peak, which sits at `r_min` along
$\theta_{\max}$, so the top of the scale is $1/r_{\min}^2$ for every panel.
The scale spans six decades. At 100 MeV the halo 15 degrees off the beam is
about a thousandth of the peak, and dividing by $r^2$ pushes it down further,
so a narrower range would turn the high-energy panels black except for the
needle itself.

`imshow` draws each map with `origin="lower"` and an `extent` that maps the
pixel grid onto the $(x, y)$ coordinates. It renders a regular grid much faster
than `pcolormesh`, which matters when five maps share one window.

Three overlays are drawn on top of each map:

- the angular pattern itself as a cyan polar curve, scaled so that its peak
  reaches 90 percent of the way to the edge,
- dashed white lines along $\theta_{\max}$, with a mirror image below the
  beam when the exit angle is zero and the pattern is symmetric,
- a gray rectangle for the target foil, a blue arrow for the incoming
  electron, and, if `exit_speed_fraction` is positive, a second arrow for
  the electron leaving the collision, with its length proportional to the
  exit speed.

Each title carries the energy, $\beta$ to six figures (so that 100 MeV does
not round to 1.0000), $\gamma$, and $\theta_{\max}$. One colorbar below the
maps serves all five. It is built from a `ScalarMappable` holding the shared
norm and colormap rather than from any one image, which states directly that
the scale belongs to the whole set of maps.

### 7.12 `main()`: the cone-angle curve

The sixth panel sweeps 300 energies from 1 keV to 1 GeV. For each it calls
`peak_angle`, so the solid curve is always the numerical peak for whatever
collision the settings describe. Two reference curves are drawn with it: the
dotted $\arccos\beta$ of Section 4, which coincides with the solid curve when
the electron stops dead, and the dashed $1/\gamma$, which joins them above
about 1 MeV. Markers show $\theta_{\max}$ and the half-energy half-angle at
the five map energies. Both axes are logarithmic, so the $1/\gamma$ law
appears as a straight line of slope $-1$ at high energy and the half-energy
markers fall along a shallower slope of $-1/2$. A horizontal line at 90
degrees marks the low-energy doughnut.

---

## 8. Reading the Plot

**10 keV and 100 keV.** Two fat lobes above and below the beam, bright at
nearly 90 degrees, dark straight ahead and straight behind. Radiation leaves
on both sides of the foil. The X-ray tube in a dentist's office works in this
regime, which is why its anode is angled to send the sideways emission out
through a window.

**1 MeV.** The lobes have swung forward to 20 degrees and the backward
hemisphere has gone dim. The forward share is already 90 percent. The
transition from "doughnut" to "cone" happens around here, where $\beta$
passes 0.9.

**10 MeV.** A narrow forward cone with $\theta_{\max}$ under 3 degrees. The
dark line exactly on the axis is still there in the mathematics, but at this
width it is a single pixel. The faint wide halo shows the logarithmic tail of
Section 5.

**100 MeV.** A needle. The cyan pattern curve is almost indistinguishable
from the axis. The halo is still there at a millionth of the peak, which is
the reason for the six-decade color scale.

**The sixth panel.** $\theta_{\max}$ sits at 90 degrees until about 100 keV,
bends over through the MeV range, and then falls as $1/\gamma$. The
half-energy markers fall more slowly, as $1/\sqrt{\gamma}$, so the gap between
"where it is brightest" and "where half the energy is" opens up with energy.

---

## 9. Things to Try

- **Add a 1 GeV electron** to `kinetic_energies_mev`. The figure has only
  five map slots, so replace one of the others. $\theta_{\max}$ drops to
  0.03 degrees and the map becomes a needle with a faint halo.
- **Deflect instead of stop.** Set `exit_speed_fraction = 1.0` and
  `exit_angle_deg = 30.0` to see a cone around each path. Then try
  `exit_angle_deg = 0.2` at 10 MeV and watch the pattern turn into the
  on-axis lobe described in Section 6.
- **Partial stop.** Set `exit_speed_fraction = 0.5` with zero exit angle.
  At 10 MeV the peak stays at 2.8 degrees but the half-energy half-angle
  drops from 17.7 to 13.5 degrees. The surviving forward term cancels the
  stopping term at wide angles, where the two are comparable, and leaves the
  narrow peak alone.
- **Compare with the Liénard pattern.** Run
  `linac_electron_radiation_direction.py` with `beta = 0.941079`, the 1 MeV
  value here. Its peak sits at about 10 degrees, this script's at about 20.
  Section 3.5 explains why one is $1/(2\gamma)$ and the other $1/\gamma$.
- **Change the cone fraction.** Call `energy_cone` with `fraction=0.9` to
  find the half-angle holding 90 percent of the energy. At 100 MeV it is
  many times $\theta_{\max}$.
- **Check the total.** Multiply the grand total inside `energy_cone` by
  $q^2/(16\pi^3\varepsilon_0 c)$ and compare with the closed form in
  Section 5.1.

---

## 10. Summary

- A charge whose velocity jumps radiates an energy per frequency per solid
  angle of $|\vec{A}(\vec{\beta}_{\text{out}}) - \vec{A}(\vec{\beta}_{\text{in}})|^2$
  times $q^2/(16\pi^3\varepsilon_0 c)$, with
  $\vec{A}(\vec{\beta}) = \hat{n}\times(\hat{n}\times\vec{\beta})/(1-\hat{n}\cdot\vec{\beta})$.
  The spectrum is flat and the pattern is the same at every frequency.
- An electron that stops dead radiates as
  $\beta^2\sin^2\theta/(1-\beta\cos\theta)^2$: nothing on the axis, thrown
  forward as $\beta \to 1$.
- The peak is at $\cos\theta_{\max} = \beta$, equivalently
  $\sin\theta_{\max} = 1/\gamma$. That is 90 degrees for a slow electron and
  $1/\gamma = m_e c^2/E$ for a fast one.
- The energy beyond the peak is spread logarithmically, so the half-energy
  cone is $\sqrt{2/\gamma}$ wide and shrinks much more slowly than the peak.
- The script draws the pattern at five energies on one shared logarithmic
  color scale, so the collapse from a sideways doughnut to a forward needle
  can be seen in a single figure, and the sixth panel tracks the cone angle
  across six decades of energy.
