# Radiation from an Electron Accelerated Along a Straight Line

Companion notes for `linac_electron_radiation_direction.py`.

This document explains the physics behind the script and then walks through
the code one piece at a time. Open it in VS Code with **Markdown: Open
Preview** (`Ctrl+Shift+V`) so the equations render.

---

## 1. The Question

Does an electron radiate when it is accelerated in a straight line, as in a
linear accelerator (LINAC)? If it does, in which direction does the radiation
go?

The short answers:

- **Yes.** *Any* accelerated charge radiates. The acceleration does not have
  to bend the path. Speeding up or slowing down along a straight line is
  enough.
- **Never straight ahead or straight behind.** The radiation leaves in a
  hollow cone around the line of motion. As the electron gets faster, the
  cone tips forward and gets narrower.

The script draws this pattern as a colored intensity map for any speed you
choose.

---

## 2. Why Accelerated Charges Radiate

A charge at rest has a static Coulomb field that falls off as $1/r^2$. A
charge moving at constant velocity carries its field along with it, and that
field still falls off as $1/r^2$. Neither one sends energy off to infinity.

When the charge **accelerates**, the change in its field cannot spread out
faster than the speed of light $c$. Close to the charge the field has already
adjusted. Far away it still reflects the old motion. The "kink" that joins the
two regions travels outward at $c$, and it is a transverse electric field
that falls off as only $1/r$.

The Poynting vector, which gives energy flow per unit area, goes as $|E|^2$.
For the kink field that is $1/r^2$. A sphere of radius $r$ has area
$4\pi r^2$, so the total power through the sphere does not depend on $r$.
That energy has left the charge for good. **That is radiation.**

The full field of a moving point charge is the **Liénard–Wiechert field**. It
has two parts:

| Part | Depends on | Falls off as | Radiates? |
|---|---|---|---|
| Velocity field | $\vec{\beta} = \vec{v}/c$ only | $1/R^2$ | No |
| Acceleration field | $\dot{\vec{\beta}} = \vec{a}/c$ | $1/R$ | **Yes** |

Only the acceleration field matters for radiation.

---

## 3. The Slow Electron: Larmor's Formula

When the electron moves slowly ($v \ll c$), the radiated power per unit solid
angle is

$$
\frac{dP}{d\Omega} = \frac{q^2 a^2}{16\pi^2 \varepsilon_0 c^3}\,\sin^2\theta
$$

where $\theta$ is the angle between the acceleration and the direction to the
observer. Integrating over all directions gives **Larmor's formula** for the
total power:

$$
P = \frac{q^2 a^2}{6\pi \varepsilon_0 c^3}
$$

The $\sin^2\theta$ factor is the important part. It is zero along the
acceleration ($\theta = 0$ and $\theta = \pi$) and largest perpendicular to
it ($\theta = \pi/2$). In three dimensions the shape is a **doughnut**
wrapped around the acceleration axis, the same pattern as a short dipole
antenna.

Where does $\sin\theta$ come from? The radiated electric field is the part of
the acceleration that is *perpendicular* to the line of sight. Light is a
transverse wave, so the component of $\vec{a}$ that points straight at the
observer cannot produce any wave toward them. The perpendicular component is
$a\sin\theta$, and the power goes as its square.

---

## 4. The Fast Electron: The Liénard Result

### 4.1 The radiation field

For a charge at any speed, the acceleration part of the Liénard–Wiechert
electric field is

$$
\vec{E}_{\text{rad}} = \frac{q}{4\pi\varepsilon_0 c}\;
\frac{\hat{n} \times \left[(\hat{n} - \vec{\beta}) \times \dot{\vec{\beta}}\right]}
{(1 - \hat{n}\cdot\vec{\beta})^3\, R}
$$

where everything on the right is evaluated at the **retarded time**, which is
when the light now reaching the observer left the charge. Here
$\hat{n}$ is the unit vector from the charge toward the observer and $R$ is
the distance between them.

### 4.2 Simplifying for straight-line acceleration

In a LINAC the acceleration points along the velocity:
$\dot{\vec{\beta}} \parallel \vec{\beta}$. Then
$\vec{\beta} \times \dot{\vec{\beta}} = 0$, and the bracket simplifies:

$$
(\hat{n} - \vec{\beta}) \times \dot{\vec{\beta}}
= \hat{n} \times \dot{\vec{\beta}} - \underbrace{\vec{\beta} \times \dot{\vec{\beta}}}_{0}
= \hat{n} \times \dot{\vec{\beta}}
$$

The numerator becomes $\hat{n} \times (\hat{n} \times \dot{\vec{\beta}})$.
Its magnitude is $\dot{\beta}\sin\theta$, which is the same transverse
projection as in the slow case. If $\theta$ is the angle between $\hat{n}$
and the direction of motion, then $\hat{n}\cdot\vec{\beta} = \beta\cos\theta$,
so

$$
|\vec{E}_{\text{rad}}| \propto \frac{a\sin\theta}{(1-\beta\cos\theta)^3\,R}
$$

### 4.3 From field to power: the exponent 5

The energy flux at the observer goes as $|E|^2$, which carries a factor of
$(1-\beta\cos\theta)^{-6}$.

There is one more correction. We want the power **the electron emits** per
unit of its own time $t'$, not the power the observer receives per unit of
the observer's time $t$. The two clocks are related by

$$
\frac{dt}{dt'} = 1 - \beta\cos\theta
$$

because the electron partly chases its own forward radiation. A pulse emitted
over a time $dt'$ arrives squeezed into the shorter time $dt$ in the forward
direction. Converting from received power to emitted power multiplies by one
factor of $(1-\beta\cos\theta)$, and the $-6$ becomes a $-5$:

$$
\boxed{\;
\frac{dP}{d\Omega} = \frac{q^2 a^2}{16\pi^2 \varepsilon_0 c^3}\,
\frac{\sin^2\theta}{(1-\beta\cos\theta)^5}
\;}
$$

**This is the Liénard result that the script plots.** Setting $\beta = 0$
brings back Larmor's $\sin^2\theta$.

### 4.4 What each factor does

| Factor | Effect |
|---|---|
| $\sin^2\theta$ | Zero exactly along the line of motion, forward and backward. Light is transverse, so this zero survives at every speed. |
| $(1-\beta\cos\theta)^{-5}$ | Large where $\cos\theta \to 1$ (forward) and small where $\cos\theta \to -1$ (backward). It tips the doughnut forward into a cone. |

The two factors pull against each other. $\sin^2\theta$ forces zero on the
axis, and the denominator piles power up as close to the axis as it can. The
result is a **hollow cone** whose half-angle shrinks as $\beta \to 1$.

---

## 5. The Angle of Peak Emission

Call the shape function $f(\theta) = \sin^2\theta\,/\,(1-\beta\cos\theta)^5$.
The peak is where $f'(\theta) = 0$:

$$
f'(\theta) = \frac{2\sin\theta\cos\theta\,(1-\beta\cos\theta) - 5\beta\sin^3\theta}
{(1-\beta\cos\theta)^6} = 0
$$

Away from $\sin\theta = 0$, divide through by $\sin\theta$ and let
$u = \cos\theta$, using $\sin^2\theta = 1 - u^2$:

$$
2u(1-\beta u) = 5\beta(1-u^2)
\quad\Longrightarrow\quad
3\beta u^2 + 2u - 5\beta = 0
$$

Take the positive root of the quadratic:

$$
\boxed{\;
\cos\theta_{\max} = \frac{\sqrt{1+15\beta^2}-1}{3\beta}
\;}
$$

The function `peak_angle()` in the script uses this formula.

### 5.1 Checking the limits

**Slow electron, $\beta \to 0$.** Expand
$\sqrt{1+15\beta^2} \approx 1 + \tfrac{15}{2}\beta^2$, which gives
$\cos\theta_{\max} \approx \tfrac{5}{2}\beta \to 0$. So
$\theta_{\max} \to 90^\circ$, which is the Larmor doughnut. The formula
divides by $\beta$, so the code treats $\beta = 0$ as a special case and
returns $\pi/2$ directly.

**Ultra-relativistic electron, $\gamma \gg 1$.** Use
$\beta^2 = 1 - 1/\gamma^2$:

$$
\sqrt{1+15\beta^2} = \sqrt{16 - 15/\gamma^2} \approx 4 - \frac{15}{8\gamma^2},
\qquad
\cos\theta_{\max} \approx 1 - \frac{1}{8\gamma^2}
$$

Comparing with $\cos\theta \approx 1 - \theta^2/2$ gives

$$
\boxed{\;\theta_{\max} \approx \frac{1}{2\gamma}\;}
$$

The script prints both the exact angle and this estimate so you can watch
them converge as $\beta$ grows.

### 5.2 How fast the cone narrows

| $\beta$ | $\gamma$ | Exact $\theta_{\max}$ | $1/(2\gamma)$ |
|---|---|---|---|
| 0.00 | 1.000 | 90.0° | (not valid) |
| 0.50 | 1.155 | 38.2° | 24.8° |
| 0.95 | 3.203 | 9.27° | 8.95° |
| 0.999 | 22.4 | 1.28° | 1.28° |

A 10 MeV electron has $\gamma \approx 20$. Its emission cone has a half-angle
of only about $1.4^\circ$.

---

## 6. Forward Versus Backward

At low speed the pattern is symmetric front to back. At high speed it is
lopsided, because $(1-\beta\cos\theta)^{-5}$ is much larger in front than
behind. The script measures this by comparing 30° with its mirror angle,
150°. The $\sin^2\theta$ factors are equal there, so the ratio is

$$
\frac{dP/d\Omega\,(30^\circ)}{dP/d\Omega\,(150^\circ)}
= \left(\frac{1+\beta\cos 30^\circ}{1-\beta\cos 30^\circ}\right)^5
$$

| $\beta$ | Ratio |
|---|---|
| 0.0 | 1 |
| 0.5 | about 103 |
| 0.95 | about 115,000 |

Even at half the speed of light, the forward direction is about one hundred
times brighter than the backward direction.

---

## 7. Total Power

Integrating the Liénard result over all solid angle gives

$$
P = \frac{q^2 a^2 \gamma^6}{6\pi\varepsilon_0 c^3}
$$

The $\gamma^6$ looks alarming, but it is misleading for a LINAC. The
companion notes `linac_electron_radiation_power.md` show that at a fixed
accelerating force the radiated power does not depend on energy at all, and
that it is only about $10^{-13}$ of the energy the electron gains.

This script plots the **shape** of the pattern, not its absolute size. For
that reason it normalizes the pattern so its peak equals 1.

---

## 8. From an Angular Pattern to a 2-D Map

The Liénard formula gives power per unit **solid angle**. To color a plane,
the script converts that to an intensity, which is power per unit **area**,
at distance $r$:

$$
I(r,\theta) = \frac{1}{r^2}\,\frac{dP}{d\Omega}(\theta)
$$

A small patch of area $dA$ at distance $r$ covers a solid angle
$d\Omega = dA / r^2$, so the same power spreads over more area farther out.

Three points about the map:

1. **Azimuthal symmetry.** The pattern depends only on $\theta$, not on the
   angle around the x-axis. Any plane that contains the line of motion
   shows the whole story. The full 3-D pattern is this picture spun around
   the x-axis.
2. **Far field.** The formula describes the radiation field, which dominates
   far from the charge. Very close to the electron the $1/r^2$ factor blows
   up and the formula no longer describes the real near field. The script
   masks out a small disk around the origin for both reasons.
3. **Directional map, not a snapshot.** The electron is moving, and its
   light takes time to travel outward. The map shows *which directions* get
   the most power, as if the emission point were fixed at the origin. It is
   not a frozen photograph of wavefronts.

---

## 9. Walking Through the Code

### 9.1 Header and imports

```python
#!/usr/bin/env -S uv run
"""linac_electron_radiation_direction.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
```

The shebang lets the script run with `uv` directly. The docstring names the
file, following the course convention. `LogNorm` gives the color bar a
logarithmic scale, which section 9.6 explains.

### 9.2 `angular_distribution(theta, beta)`

```python
pattern = np.sin(theta) ** 2 / (1 - beta * np.cos(theta)) ** 5
return pattern / peak_value(beta)
```

This is the Liénard shape function $f(\theta)$, divided by its peak value so
the result runs from 0 to 1. The constant $q^2 a^2 / (16\pi^2\varepsilon_0 c^3)$
is left out because it only changes the overall scale. Because the function
uses NumPy operations, `theta` can be a single number or a whole 2-D grid.

### 9.3 `peak_angle(beta)`

```python
if beta == 0:
    return np.pi / 2
return float(np.arccos((np.sqrt(1 + 15 * beta**2) - 1) / (3 * beta)))
```

This is the closed-form result from section 5. The $\beta = 0$ branch avoids
dividing by zero and returns the Larmor answer of $90^\circ$.

### 9.4 `peak_value(beta)`

This function evaluates the unnormalized $f(\theta)$ at $\theta_{\max}$. It
is the normalizing constant used in `angular_distribution()`. Because the peak
position is known exactly, the normalization does not depend on how fine the
plotting grid is.

### 9.5 `main()`: the physics summary

```python
# TODO: Change the electron speed (fraction of c, 0 <= beta < 1)
beta = 0.5
gamma = 1 / np.sqrt(1 - beta**2)
```

**The `beta` line is the one to edit.** The script then computes $\gamma$,
the exact $\theta_{\max}$, and the 30° versus 150° ratio, and prints them.
The $1/(2\gamma)$ line is only meaningful when $\gamma \gg 1$. At $\beta = 0$
it prints 28.6°, which is a reminder of where the approximation fails.

### 9.6 `main()`: building the grid

```python
x, y = np.meshgrid(
    np.linspace(-extent, extent, n_points),
    np.linspace(-extent, extent, n_points),
)
r = np.hypot(x, y)
theta = np.arctan2(np.abs(y), x)
```

`meshgrid` makes an $801 \times 801$ grid of points covering
$[-1, 1] \times [-1, 1]$. For each point:

- `np.hypot(x, y)` gives the distance $r = \sqrt{x^2+y^2}$.
- `np.arctan2(np.abs(y), x)` gives the angle from the $+x$ axis, which is the
  direction of motion. Taking `np.abs(y)` folds the lower half-plane onto the
  upper half, so $\theta$ always lies between $0$ and $\pi$. That is the range
  the Liénard formula expects, and it makes the symmetry about the x-axis
  automatic.

```python
r_min = 0.04
intensity = angular_distribution(theta, beta) / np.maximum(r, r_min) ** 2
intensity = np.ma.masked_where(r < r_min, intensity)
```

This applies $I = f(\theta)/r^2$. `np.maximum(r, r_min)` keeps the division
from hitting zero at the origin. `masked_where` then hides the small central
disk, which `pcolormesh` leaves blank.

```python
i_max = intensity.max()
norm = LogNorm(vmin=i_max * 1e-4, vmax=i_max)
intensity = np.clip(intensity, norm.vmin, None)
```

**Why a log scale?** The intensity covers many orders of magnitude. The
$1/r^2$ factor alone gives a factor of several hundred across the plot, and
at high $\beta$ the forward/backward contrast adds a factor of $10^5$ or more.
On a linear scale almost everything would be black. The log scale shows four
decades. Values below that floor are clipped to the floor rather than masked,
so the axis zeros show as solid dark instead of blank holes.

### 9.7 `main()`: drawing

```python
mesh = plt.pcolormesh(x, y, intensity, cmap="inferno", norm=norm, shading="auto")
```

`pcolormesh` colors each grid cell by its intensity. The `inferno` colormap
runs from black through red and orange to pale yellow. It is perceptually
uniform, so equal steps in brightness mean equal steps in log intensity.

**Polar overlay.**

```python
phi = np.linspace(-np.pi, np.pi, 2_000)
lobe = 0.9 * extent * angular_distribution(np.abs(phi), beta)
plt.plot(lobe * np.cos(phi), lobe * np.sin(phi), ...)
```

This is a classic antenna-pattern polar plot. For each direction $\varphi$,
the code draws a point at distance proportional to $dP/d\Omega$. The cyan
curve shows the lobes directly, on a *linear* scale, which complements the
log-scale colors.

**Peak-angle lines.** Two white dashed lines run from the origin at
$\pm\theta_{\max}$. Only the upper one gets a legend label, so the legend
does not list it twice.

**Electron and arrow.** A dot at the origin marks the electron.
`plt.annotate` with an empty string and `arrowprops` draws an arrow along
$+x$, labeled $\vec{v}, \vec{a}$ to show that the velocity and acceleration
are parallel.

**Finishing touches.** `set_aspect("equal")` keeps angles true, so a
$38^\circ$ line actually looks like $38^\circ$. The figure window is named
after the script with `Path(__file__).name`, following the course
convention.

---

## 10. Reading the Plot

| What you see | Physics behind it |
|---|---|
| Dark streak along the +x and −x axes | $\sin^2\theta = 0$: no radiation straight ahead or straight behind |
| Two bright lobes at $\pm\theta_{\max}$ | The hollow emission cone, seen in cross-section |
| Lobes tilting forward as $\beta$ grows | $(1-\beta\cos\theta)^{-5}$ concentrating power forward |
| Brightness falling with distance | $1/r^2$ spreading of a fixed solid angle |
| Very dim left half at high $\beta$ | Backward emission suppressed by the Doppler-like factor |

---

## 11. Things to Try

1. **Set `beta = 0.0`.** You should see a symmetric doughnut with
   $\theta_{\max} = 90^\circ$ and a forward/backward ratio of exactly 1.
2. **Increase `beta` through 0.3, 0.7, 0.9, 0.99.** Watch the lobes swing
   forward. Compare the printed $\theta_{\max}$ with $1/(2\gamma)$.
3. **Try `beta = 0.999`.** The lobes become needle-thin. You may need to raise
   `n_points` or zoom in with the Matplotlib toolbar to resolve them.
4. **Change the color floor** from `1e-4` to `1e-8`. Faint backward emission
   becomes visible, but the map loses contrast elsewhere.
5. **Try a linear scale.** Replace `norm=norm` with no norm, and remove the
   clipping. This shows why the log scale was chosen.
6. **Challenge: received power.** Change the exponent from 5 to 6 to plot the
   power the observer *receives* per unit of their own time, and compare
   $\theta_{\max}$. Derive the new peak-angle formula yourself.

---

## 12. Summary

- A charge accelerated along a straight line **does** radiate.
- The radiation leaves in a **hollow cone**, with zero intensity exactly
  along the line of motion.
- The Liénard formula $dP/d\Omega \propto \sin^2\theta\,/\,(1-\beta\cos\theta)^5$
  combines the transverse nature of light, $\sin^2\theta$, with relativistic
  forward beaming, $(1-\beta\cos\theta)^{-5}$.
- The cone half-angle is $\cos\theta_{\max} = (\sqrt{1+15\beta^2}-1)/(3\beta)$,
  which approaches $1/(2\gamma)$ at high energy.
- `linac_electron_radiation_direction.py` turns this into a log-scaled
  intensity map with a polar-plot overlay. You can rerun it at any speed by
  changing one line.
