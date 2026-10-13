# Synchrotron Radiation: Where It Goes

Companion notes for `synchrotron_radiation_direction.py`.

This document explains the physics behind the script and then walks through
the code one piece at a time. Open it in VS Code with **Markdown: Open
Preview** (`Ctrl+Shift+V`) so the equations render.

The LINAC documents in this folder treat an electron whose acceleration is
*parallel* to its velocity. This document treats the other case: an electron
on a circular orbit, whose acceleration points at the center of the ring, at
right angles to its velocity. The radiation it emits is called **synchrotron
radiation**. Its total power is the subject of `synchrotron_radiation_power.md`.
This document is about its direction.

---

## 1. The Question

An electron circulates in a storage ring. In which directions does it
radiate, and how does the pattern change with speed?

The short answers:

- **Straight ahead.** A fast electron radiates a headlight beam along its
  velocity, of half-angle about $1/\gamma$, and almost nothing anywhere
  else. The beam is tangent to the ring, so as the electron goes round the
  beam sweeps the plane like a lighthouse.
- **With two dark lines in the orbit plane.** The pattern is exactly zero at
  the angles $\theta = \pm\arcsin(1/\gamma)$ from the velocity, measured in
  the plane of the orbit. There are no such nulls out of the plane.
- **Not hollow.** This is the opposite of the LINAC case. Acceleration along
  the velocity makes a cone with a dark axis; acceleration across it makes a
  beam that is brightest on the axis.

The script draws the pattern in the orbit plane as a map around the ring,
draws what a screen facing the oncoming electron would see, and can animate
the sweep.

---

## 2. The Setup

The orbit lies in the $x$-$y$ plane, a circle of radius $R$ centered on the
origin, traversed counterclockwise. At the moment of the snapshot the electron
is at the bottom of the ring, moving in the $+x$ direction. Its acceleration
points up, toward the center. In the formulas below the velocity direction is
the $x$ axis and the acceleration direction is the $y$ axis:

$$
\vec{\beta} = \beta\,\hat{x}, \qquad \hat{a} = \hat{y}.
$$

A direction of observation $\hat{n}$ is described by the angle $\theta$ from
the velocity and the azimuth $\phi$ around it, measured from the
acceleration, so that

$$
\hat{n} = (\cos\theta,\ \sin\theta\cos\phi,\ \sin\theta\sin\phi).
$$

$\phi = 0$ and $\phi = \pi$ are the orbit plane. $\phi = \pm\pi/2$ is the
plane perpendicular to the orbit that contains the velocity.

---

## 3. The Liénard Pattern for Perpendicular Acceleration

### 3.1 The general formula

`linac_electron_radiation_direction.md` builds the radiation field of a
moving charge and arrives at the power radiated per unit solid angle, for any
velocity and any acceleration:

$$
\frac{dP}{d\Omega}
= \frac{q^2}{16\pi^2\varepsilon_0 c}\,
\frac{\big|\hat{n}\times\big[(\hat{n}-\vec{\beta})\times\dot{\vec{\beta}}\big]\big|^2}
{(1-\hat{n}\cdot\vec{\beta})^5},
\qquad \dot{\vec{\beta}} = \frac{\vec{a}}{c}.
$$

The script's `lienard_pattern` function evaluates the vector expression
directly, for any arrangement of the three vectors, which is what the screen
panel and the integral over the sphere use.

### 3.2 Reducing it for a circular orbit

With $\vec{\beta} = \beta\hat{x}$ and $\dot{\vec{\beta}} = (a/c)\hat{y}$ the
triple product can be expanded by hand. Using
$\vec{A}\times(\vec{B}\times\vec{C}) = \vec{B}(\vec{A}\cdot\vec{C}) - \vec{C}(\vec{A}\cdot\vec{B})$
with $\vec{A} = \hat{n}$, $\vec{B} = \hat{n}-\vec{\beta}$, $\vec{C} = \hat{y}$,

$$
\hat{n}\times\big[(\hat{n}-\vec{\beta})\times\hat{y}\big]
= (\hat{n}\cdot\hat{y})\,(\hat{n}-\vec{\beta}) - (1-\hat{n}\cdot\vec{\beta})\,\hat{y}.
$$

Now $\hat{n}\cdot\hat{y} = \sin\theta\cos\phi$ and
$\hat{n}\cdot\vec{\beta} = \beta\cos\theta$. Squaring, with
$|\hat{n}-\vec{\beta}|^2 = 1 - 2\beta\cos\theta + \beta^2$ and
$(\hat{n}-\vec{\beta})\cdot\hat{y} = \sin\theta\cos\phi$,

$$
\big|\cdots\big|^2
= \sin^2\theta\cos^2\phi\,(1 - 2\beta\cos\theta + \beta^2)
- 2\sin^2\theta\cos^2\phi\,(1-\beta\cos\theta)
+ (1-\beta\cos\theta)^2 .
$$

The first two terms combine to $-\sin^2\theta\cos^2\phi\,(1-\beta^2)$, and
$1-\beta^2 = 1/\gamma^2$, so

$$
\big|\cdots\big|^2
= (1-\beta\cos\theta)^2\left[1 - \frac{\sin^2\theta\cos^2\phi}{\gamma^2(1-\beta\cos\theta)^2}\right].
$$

Dividing by $(1-\beta\cos\theta)^5$ gives the result in the header of the
script:

$$
\boxed{\;
\frac{dP}{d\Omega}
= \frac{q^2 a^2}{16\pi^2\varepsilon_0 c^3}\,
\frac{1}{(1-\beta\cos\theta)^3}
\left[1 - \frac{\sin^2\theta\cos^2\phi}{\gamma^2(1-\beta\cos\theta)^2}\right]
\;}
$$

The script checks this reduction numerically. It evaluates both the vector
form and the closed form around the orbit plane and prints the largest
difference, which is at the level of rounding error.

---

## 4. Reading the Formula

### 4.1 Straight ahead and straight behind

On the axis $\sin\theta = 0$, the bracket is 1, and the pattern is
$1/(1-\beta\cos\theta)^3$. Straight ahead that is $1/(1-\beta)^3$, which for
a fast electron is about $8\gamma^6$. Straight behind it is $1/(1+\beta)^3$.
The ratio

$$
\left(\frac{1+\beta}{1-\beta}\right)^3
$$

is 6859 at $\beta = 0.9$ and about $64\gamma^6$ for $\beta \to 1$. The
pattern is **brightest exactly along the velocity**, with no dark axis. All
of the plots are normalized to this forward value.

### 4.2 The nulls in the orbit plane

In the orbit plane $\cos^2\phi = 1$ and the bracket is

$$
1 - \frac{\sin^2\theta}{\gamma^2(1-\beta\cos\theta)^2},
$$

which vanishes where $\sin\theta = \gamma\,(1-\beta\cos\theta)$. Rearranged,
$\sin\theta + \gamma\beta\cos\theta = \gamma$. The left side is
$\sqrt{1+\gamma^2\beta^2}\,\sin(\theta+\alpha)$ with $\tan\alpha = \gamma\beta$,
and $1+\gamma^2\beta^2 = \gamma^2$, so the equation reads
$\sin(\theta+\alpha) = 1$, which means $\theta = \pi/2 - \alpha$. Then
$\tan\theta_{\text{null}} = 1/(\gamma\beta) = 1/\sqrt{\gamma^2-1}$, or more
simply

$$
\boxed{\;\sin\theta_{\text{null}} = \frac{1}{\gamma}\;}
$$

For a slow electron that is $90^\circ$: the dipole null along the
acceleration. For a fast one it is $1/\gamma$. Between the two nulls lies the
headlight. Beyond them, further from the axis, the bracket is positive again
and there are weak side lobes.

This is the same angle at which the bremsstrahlung pattern of
`linac_electron_bremsstrahlung_direction.md` *peaks*. The coincidence has a
simple source. Where $\sin\theta = 1/\gamma$, also $\cos\theta = \beta$, so
the Doppler factor $1 - \beta\cos\theta$ equals exactly $1 - \beta^2 = 1/\gamma^2$.
Both results come from a competition between that factor and a
$\sin\theta$, and both single out the angle where it takes this value.

### 4.3 Out of the orbit plane

In the plane perpendicular to the orbit, $\cos\phi = 0$, the bracket is 1
and the pattern is just $1/(1-\beta\cos\theta)^3$. It falls off smoothly with
no nulls. For small angles $1 - \beta\cos\theta \approx (1+\gamma^2\theta^2)/(2\gamma^2)$,
so the pattern goes as $1/(1+\gamma^2\theta^2)^3$ and drops to half its peak
at $\gamma\theta = 0.51$.

### 4.4 Small angles in the orbit plane

With the same approximation the in-plane pattern becomes

$$
\frac{dP}{d\Omega} \propto \frac{(1-\gamma^2\theta^2)^2}{(1+\gamma^2\theta^2)^5},
$$

which shows the nulls at $\gamma\theta = \pm 1$ as double zeros and the side
lobes beyond them. On the screen panel of the plot these are the two crosses
on the horizontal axis and the two faint wings outside them.

### 4.5 The slow limit

As $\beta \to 0$ the bracket is $1 - \sin^2\theta\cos^2\phi$. Since
$\sin\theta\cos\phi$ is the cosine of the angle between $\hat{n}$ and the
acceleration, this is $\sin^2$ of that angle: the dipole doughnut with its
axis along $\vec{a}$. In the orbit plane that is a figure-eight along the
velocity line, dark toward and away from the center of the ring.

### 4.6 Parallel versus perpendicular

| | Acceleration along $\vec{v}$ (LINAC) | Acceleration across $\vec{v}$ (ring) |
| --- | --- | --- |
| Pattern | $\dfrac{\sin^2\theta}{(1-\beta\cos\theta)^5}$ | $\dfrac{1}{(1-\beta\cos\theta)^3}\left[1 - \dfrac{\sin^2\theta\cos^2\phi}{\gamma^2(1-\beta\cos\theta)^2}\right]$ |
| On the axis | Zero | Maximum |
| Brightest at | $\theta \approx 1/(2\gamma)$ | $\theta = 0$ |
| Nulls | The axis, at every $\phi$ | $\sin\theta = 1/\gamma$, in the orbit plane only |
| Total power | $\propto \gamma^6 a^2$ | $\propto \gamma^4 a^2$ |

---

## 5. Total Power and the Cone

Integrating the pattern over all directions gives

$$
P = \frac{q^2 a^2 \gamma^4}{6\pi\varepsilon_0 c^3} .
$$

In the units of the script, where the pattern is
$|\cdots|^2/(1-\hat{n}\cdot\vec{\beta})^5$ without the constant, this says
the integral over the sphere equals $\tfrac{8\pi}{3}\gamma^4$. The script
integrates numerically and prints the ratio, which is 1.00000.

The script also prints the share of the power inside the cone
$\theta < 1/\gamma$. At $\beta = 0.9$ it is 74 percent. The rest is in the
wings and side lobes. Unlike the bremsstrahlung case, where the energy was
spread logarithmically far outside the peak, synchrotron radiation really is
confined to a cone of order $1/\gamma$.

The comparison with the LINAC is sharpest at a fixed *force*. Along the
velocity $dp/dt = \gamma^3 m a$, across it $dp/dt = \gamma m a$. For the same
$|dp/dt|$ the perpendicular acceleration is $\gamma^2$ larger, so the
perpendicular power, $\propto \gamma^4 a^2$, exceeds the parallel power,
$\propto \gamma^6 a^2$, by a factor $\gamma^2$. Bending a beam costs far more
than pushing it.

---

## 6. The Lighthouse Flash

The headlight is tangent to the orbit and turns with the electron. An observer
sitting in the orbit plane is lit only while the beam, of full width about
$2/\gamma$, sweeps across the line of sight. The electron covers that angle
in a time $2R/(\gamma c)$ on its own clock, and the radiation it emits during
that time arrives compressed by the Doppler factor $1 - \beta \approx 1/(2\gamma^2)$.
The flash the observer sees lasts

$$
\Delta t \approx \frac{2R}{\gamma c}\cdot\frac{1}{2\gamma^2} = \frac{R}{\gamma^3 c} .
$$

A pulse that short contains frequencies up to about $\gamma^3 c/R$, which is
the origin of the critical frequency $\omega_c = \tfrac{3}{2}\gamma^3 c/R$ in
the power document. For a 3 GeV electron ($\gamma = 5900$) on a 25 m radius
the flash lasts $4 \times 10^{-19}$ s and the spectrum reaches into the
X-rays. Every synchrotron light source is built on this fact. The
`animate = True` setting shows the sweep.

---

## 7. Walking Through the Code

### 7.1 Header and imports

```python
#!/usr/bin/env -S uv run
"""synchrotron_radiation_direction.py"""
```

`FuncAnimation` drives the optional animation, `LogNorm` gives the maps
their logarithmic color scale, `Circle` draws the ring, and `trapezoid` does
the integral over the sphere.

### 7.2 `lienard_pattern(n, beta_vec, accel_dir)`

The general vector formula of Section 3.1 without its constant. `n` is an
array of unit vectors with shape `(..., 3)`, so one call evaluates the
pattern on a whole grid. The two cross products use `np.cross`, which
broadcasts a single 3-vector against the array. The result is
$|\hat{n}\times((\hat{n}-\vec{\beta})\times\hat{a})|^2 / (1-\hat{n}\cdot\vec{\beta})^5$.

### 7.3 `in_plane_pattern(theta, beta)`

The closed form of Section 3.2 for the orbit plane, where $\cos^2\phi = 1$.
It gives the same numbers as `lienard_pattern` restricted to the plane but
runs about five times faster, because it avoids building vectors and taking
cross products. That matters for the animation, which redraws the map on
every frame.

### 7.4 `null_angle(beta)`

Returns $\arcsin\sqrt{1-\beta^2} = \arcsin(1/\gamma)$, the result of
Section 4.2.

### 7.5 `power_fractions(beta)`

Lays a grid over the sphere, 4001 polar angles by 181 azimuths, evaluates
the vector formula at every point, and integrates with the solid-angle weight
$\sin\theta$. It returns the total and the fraction inside $\theta < 1/\gamma$.
The total is compared with $\tfrac{8\pi}{3}\gamma^4$ in the printout.

### 7.6 `orbit_plane_intensity(x, y, position, velocity_dir, beta)`

Colors every point of the plane by the pattern along the line from the
electron to that point. The angle from the velocity comes from a two-
dimensional cross and dot product with `np.arctan2`, and the sign is
irrelevant because the in-plane pattern is symmetric about the velocity.

This map deliberately leaves out the $1/r^2$ falloff that the LINAC maps
include. With it, a narrow beam fades within a short distance of the
electron and the picture is mostly black. Without it, the headlight shows as
a wedge of uniform brightness, with the two null lines as crisp dark edges,
which is the lighthouse image the script is after. The colorbar label says
what is plotted.

### 7.7 `main()`: the settings and the checks

```python
# TODO: Change the electron speed (fraction of c, 0 <= beta < 1)
beta = 0.9

# TODO: Set True to watch the electron circulate and the beam sweep the ring
animate = False
```

The printout lists $\beta$, $\gamma$, the null angle beside $1/\gamma$, the
forward-to-backward ratio, the sphere integral divided by $\tfrac{8\pi}{3}\gamma^4$,
the share inside the cone, and the agreement between the closed form and the
vector formula.

### 7.8 `main()`: the orbit plane

Every artist that moves with the electron is created once, empty, and then
filled in by `update(frame)`: the image, the electron dot, the cyan polar
overlay, the two dashed null lines, and the velocity and acceleration arrows
with their labels. `update` places the electron at ring angle
$-90^\circ + 2^\circ \times$ `frame`, builds the tangent and the inward
radial direction, recomputes the map, and repositions everything. For the
static figure `update(0)` is called once.

The polar overlay is computed once as a radius-versus-angle curve, scaled so
that its forward peak is 0.45 units long, a little less than the ring
radius, and only rotated and translated in `update`.

### 7.9 `main()`: the screen

The right panel evaluates the vector formula on a grid of directions
$\hat{n} = (\cos h\cos v,\ \sin h\cos v,\ \sin v)$, where $h$ is the angle in
the orbit plane and $v$ the angle out of it, both in units of $1/\gamma$. The
half-width of the view is $\min(4,\ 1.5\gamma)$ so that a slow electron, whose
pattern is $90^\circ$ wide, does not wrap around. A dashed circle marks the
cone $\theta = 1/\gamma$, a dotted line marks the orbit plane, and two
crosses sit on the nulls at $h = \pm\gamma\,\theta_{\text{null}}$.

### 7.10 `main()`: the animation

With `animate = True`, `FuncAnimation` calls `update` for 180 frames at 40 ms
intervals, one full turn at $2^\circ$ per frame. Each frame recomputes the
601 by 601 map in about 35 ms. The animation object is attached to the figure
so that it is not garbage collected before the window closes.

---

## 8. Reading the Plot

**Left panel.** The dashed circle is the ring, the blue dot the electron at
its bottom, moving right. The bright wedge leaving it to the right is the
headlight, tangent to the ring. Its two edges are the exact nulls at
$\pm 25.8^\circ$ for $\beta = 0.9$, drawn dashed. Outside the nulls the plane
is dim but not black: those are the side lobes, a few percent of the peak
near the nulls and falling to $10^{-4}$ behind the electron. The cyan curve
is the same pattern as a polar plot, mostly a long forward lobe.

**Right panel.** The bright central spot is the headlight seen head-on. The
dashed circle of radius 1 is the $1/\gamma$ cone and the spot roughly fills
it. On the horizontal axis, the orbit plane, the spot is pinched to zero at
the two crosses and faint wings appear beyond them. Vertically there is no
pinch, only a smooth fade.

**Changing $\beta$.** At $\beta = 0.5$ the nulls are at $60^\circ$ and the
backward lobe is still visible. At $\beta = 0.99$ the wedge is $16^\circ$
wide and the backward hemisphere is black to the bottom of the color scale.

---

## 9. Things to Try

- **Slow it down.** Set `beta = 0.01`. The orbit-plane map becomes a
  figure-eight along the velocity with nulls at $90^\circ$, toward and away
  from the center of the ring, and the screen shows a broad dipole pattern.
- **Speed it up.** Set `beta = 0.99`. The null angle drops to $8.1^\circ$ and
  the share of the power inside the $1/\gamma$ cone changes; compare it with
  the value at 0.9.
- **Animate.** Set `animate = True` and watch the beam sweep. Imagine
  standing at a fixed point in the plane and note how briefly the beam
  crosses you.
- **Compare with the LINAC.** Run `linac_electron_radiation_direction.py`
  with the same $\beta$ and set the two windows side by side: dark axis
  versus bright axis, $1/(2\gamma)$ versus $1/\gamma$.
- **Reverse the ring.** Change the sign of `velocity_dir` in `update` so the
  electron runs clockwise and confirm the beam still leaves tangent to the
  ring, in the new direction of motion.

---

## 10. Summary

- For acceleration perpendicular to the velocity the Liénard pattern is
  $[1 - \sin^2\theta\cos^2\phi/(\gamma^2(1-\beta\cos\theta)^2)]/(1-\beta\cos\theta)^3$.
- It is brightest straight ahead, by $8\gamma^6$ relative to the
  nonrelativistic scale, and confined to a cone of half-angle about
  $1/\gamma$.
- In the orbit plane it has exact nulls at $\sin\theta = 1/\gamma$, with weak
  side lobes beyond. Out of the plane it fades smoothly.
- The total power is $q^2 a^2\gamma^4/(6\pi\varepsilon_0 c^3)$, which the
  script confirms by integrating the pattern over the sphere.
- The beam is tangent to the ring and sweeps with the electron, so a fixed
  observer sees a flash of duration $R/(\gamma^3 c)$, which is why the
  spectrum reaches the X-ray region.
