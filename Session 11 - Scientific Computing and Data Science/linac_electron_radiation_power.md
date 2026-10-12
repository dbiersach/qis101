# How Much Does a LINAC Electron Radiate?

Companion notes for `linac_electron_radiation_power.py`.

This document explains the physics behind the script and then walks through
the code one piece at a time. Open it in VS Code with **Markdown: Open
Preview** (`Ctrl+Shift+V`) so the equations render. The *direction* of the
radiation is covered in `linac_electron_radiation_direction.md`. This
document is about how *much* power is radiated.

---

## 1. The Puzzle

Integrating the Liénard angular distribution over all directions gives the
total power radiated by a charge whose acceleration is parallel to its
velocity:

$$
\textbf{Form 1:}\qquad
P = \frac{q^2 a^2 \gamma^6}{6\pi\varepsilon_0 c^3}
$$

Here $a = dv/dt$ is the ordinary acceleration and
$\gamma = 1/\sqrt{1-v^2/c^2}$ is the Lorentz factor.

**At first glance this is alarming.** A 10 GeV electron has
$\gamma \approx 20{,}000$, and $\gamma^6 \approx 6 \times 10^{25}$. It looks
as if a high-energy LINAC should radiate away everything it puts in.

The same power can also be written as

$$
\textbf{Form 2:}\qquad
P = \frac{q^2}{6\pi\varepsilon_0 m^2 c^3}\left(\frac{dp}{dt}\right)^2
$$

There is no $\gamma$ anywhere in it. At a fixed accelerating force the power
is the **same at every energy**.

Both forms are correct. The script evaluates them side by side to show that
they always agree, and that the $\gamma^6$ is exactly cancelled.

---

## 2. Why the Two Forms Are the Same

### 2.1 Relativistic momentum

The momentum of a relativistic particle is

$$
p = \gamma m v
$$

Both $\gamma$ and $v$ change as the particle speeds up. Differentiate with
the product rule:

$$
\frac{dp}{dt} = m\left(\gamma\,\frac{dv}{dt} + v\,\frac{d\gamma}{dt}\right)
$$

### 2.2 How fast $\gamma$ changes

Differentiate $\gamma = (1 - v^2/c^2)^{-1/2}$ with the chain rule:

$$
\frac{d\gamma}{dt}
= -\tfrac{1}{2}\left(1-\frac{v^2}{c^2}\right)^{-3/2}\left(-\frac{2v}{c^2}\,\frac{dv}{dt}\right)
= \gamma^3\,\frac{v}{c^2}\,a
$$

### 2.3 Putting it together

Substitute this back in:

$$
\frac{dp}{dt} = m a\left(\gamma + \gamma^3\frac{v^2}{c^2}\right)
= m a\,\gamma\left(1 + \gamma^2\beta^2\right)
$$

Since $\gamma^2 = 1/(1-\beta^2)$, we have
$1 + \gamma^2\beta^2 = \gamma^2$, so

$$
\boxed{\;\frac{dp}{dt} = \gamma^3\, m\, a\qquad\text{(force parallel to velocity)}\;}
$$

Square this and substitute $a^2\gamma^6 = (dp/dt)^2/m^2$ into Form 1. The
result is exactly Form 2. **The two forms are algebraically identical.**

### 2.4 What this means physically

The factor $\gamma^3 m$ is sometimes called the **longitudinal mass**. It is
the resistance a fast particle shows to being pushed along its direction of
motion. As the electron nears $c$, a fixed force produces less and less
acceleration, because its speed cannot exceed $c$. The acceleration falls as
$\gamma^{-3}$, so $a^2$ falls as $\gamma^{-6}$, which cancels the $\gamma^6$
in Form 1 exactly.

**Form 1 asks:** "how hard is the electron's speed changing?" At high energy
the answer is "hardly at all", multiplied by a huge $\gamma^6$.

**Form 2 asks:** "how hard is the electron being pushed?" In a LINAC that
answer never changes.

### 2.5 A Lorentz-invariant view (optional)

The general relativistic Larmor formula can be written in terms of the
four-momentum $p^\mu = (E/c,\,\vec{p})$ and proper time $\tau$:

$$
P = \frac{q^2}{6\pi\varepsilon_0 m^2 c^3}
\left[\left(\frac{d\vec{p}}{d\tau}\right)^2 - \frac{1}{c^2}\left(\frac{dE}{d\tau}\right)^2\right]
$$

For straight-line motion $dE/dt = v\,dp/dt$, and $d\tau = dt/\gamma$, so the
bracket becomes

$$
\gamma^2\left(\frac{dp}{dt}\right)^2\left(1 - \beta^2\right) = \left(\frac{dp}{dt}\right)^2
$$

That is Form 2 again. Because the radiated power is a Lorentz invariant, a
form written in terms of force is the natural one to use.

---

## 3. Applying It to a LINAC

### 3.1 A constant accelerating force

A LINAC pushes electrons with an electric field $\mathcal{E}$ in its
accelerating cavities. The field is roughly the same all along the machine,
and it is usually quoted as an **accelerating gradient** in MeV per meter. A
typical value for a normal-conducting copper structure is about 20 MeV/m.

For a particle of charge $e$, a gradient of $G$ MeV/m means the field is $G$
MV/m, so

$$
F = \frac{dp}{dt} = e\,\mathcal{E}
$$

The force is the same at every electron energy. In the script it is

$$
F = (1.602 \times 10^{-19}\ \text{C})(2.0\times10^{7}\ \text{V/m})
= 3.20\times 10^{-12}\ \text{N}
$$

### 3.2 The radiated power

Put this into Form 2:

$$
P = \frac{e^2 F^2}{6\pi\varepsilon_0 m_e^2 c^3} \approx 7.06 \times 10^{-17}\ \text{W}
$$

That is about seventy *attowatts*, at every energy from 0.5 MeV to 10 GeV.

### 3.3 Radiated power compared with energy gain

The useful question is not "how many watts" but "what fraction of the energy
gained is lost". The work-energy theorem gives the rate of energy gain:

$$
\frac{dE}{dt} = F v = F\beta c
$$

Divide the radiated power by this rate:

$$
\frac{P_{\text{rad}}}{dE/dt}
= \frac{e^2 F^2 / (6\pi\varepsilon_0 m^2 c^3)}{F\beta c}
= \frac{e^2 F}{6\pi\varepsilon_0 m^2 c^4\,\beta}
$$

Now introduce the **classical electron radius**,

$$
r_e = \frac{e^2}{4\pi\varepsilon_0 m c^2} \approx 2.818 \times 10^{-15}\ \text{m}
$$

This is the radius at which the electrostatic energy of a ball of charge $e$
equals $mc^2$. Using it, and noting that $F = dE/dx$ is the energy gained per
meter:

$$
\boxed{\;
\frac{P_{\text{rad}}}{dE/dt} = \frac{2}{3}\,\frac{r_e}{m c^2}\,\frac{dE/dx}{\beta}
\;}
$$

For $\beta \to 1$ this is the estimate printed by the script:

$$
\frac{2}{3}\times
\frac{2.818\times10^{-15}\ \text{m}}{0.511\ \text{MeV}}
\times 20\ \frac{\text{MeV}}{\text{m}}
\approx 7.35 \times 10^{-14}
$$

**The electron loses less than one part in ten trillion of the energy it
gains.** Another way to say it: over each meter the electron gains 20 MeV and
radiates about $1.5 \times 10^{-6}$ eV.

The $1/\beta$ explains the first row of the script's table. At 0.59 MeV the
electron has $\beta \approx 0.51$, so the fraction is about twice the
high-energy value. Energy gain is slower there because the electron moves
slower, while the radiated power stays the same.

### 3.4 When would radiation matter?

Set the loss fraction equal to 1 and solve for the gradient:

$$
\frac{dE}{dx} \sim \frac{3}{2}\,\frac{mc^2}{r_e}
\approx 3 \times 10^{14}\ \text{MeV/m}
$$

That is about $10^{13}$ times stronger than any real accelerator. At fields
long before that, near the **Schwinger limit** of about
$1.3\times10^{18}$ V/m ($1.3\times10^{12}$ MeV/m), classical electrodynamics
itself breaks down and quantum effects take over. For straight-line
acceleration, classical radiation losses never matter in practice.

---

## 4. Contrast: Circular Machines

The picture changes completely when the force is **perpendicular** to the
velocity, as in a ring. Then $|\vec{v}|$ and $\gamma$ stay fixed, and the
momentum changes only in direction:

$$
\frac{dp}{dt} = \gamma m a \qquad\text{(force perpendicular to velocity)}
$$

This has $\gamma^1$, not $\gamma^3$. The Liénard result for this case is

$$
P_\perp = \frac{q^2 a^2\gamma^4}{6\pi\varepsilon_0 c^3}
= \frac{q^2\gamma^2}{6\pi\varepsilon_0 m^2 c^3}\left(\frac{dp}{dt}\right)^2
$$

For the **same force**, a perpendicular push radiates $\gamma^2$ times more
than a parallel push. In a ring of bending radius $\rho$ the centripetal
acceleration is $a = v^2/\rho \approx c^2/\rho$, so

$$
P_\perp \propto \frac{\gamma^4}{\rho^2}
$$

The energy lost per turn is commonly written

$$
U_0\,[\text{keV}] \approx 88.5\,\frac{E^4\,[\text{GeV}]}{\rho\,[\text{m}]}
$$

CERN's LEP collider ran electrons at about 100 GeV with $\rho \approx
3100$ m. That gives

$$
U_0 \approx 88.5 \times \frac{(100)^4}{3100}\ \text{keV} \approx 2.9\ \text{GeV per turn}
$$

That is about 3% of the beam energy, lost on every single lap, and it had to
be replaced by RF cavities. This $\gamma^4$ wall is a main reason future
high-energy electron colliders are planned as **linear** machines.

| | LINAC (parallel force) | Ring (perpendicular force) |
|---|---|---|
| $dp/dt$ vs. $a$ | $\gamma^3 m a$ | $\gamma m a$ |
| $P$ at fixed $dp/dt$ | independent of $\gamma$ | grows as $\gamma^2$ |
| $P$ in practice | about $10^{-16}$ W | megawatts for a whole beam |

---

## 5. Walking Through the Code

### 5.1 Header and imports

```python
from scipy.constants import c, e, epsilon_0, m_e, physical_constants
```

All physical constants come from `scipy.constants`, which uses CODATA
values, so nothing is typed in by hand. `physical_constants` is a dictionary
that also provides the classical electron radius $r_e$.

### 5.2 `power_from_acceleration(a, gamma)`

```python
return e**2 * a**2 * gamma**6 / (6 * np.pi * epsilon_0 * c**3)
```

This is Form 1, written exactly as it appears in section 1. It takes arrays,
so a single call evaluates the power at all 400 energies.

### 5.3 `power_from_force(dp_dt)`

```python
return e**2 * dp_dt**2 / (6 * np.pi * epsilon_0 * m_e**2 * c**3)
```

This is Form 2. The two functions are written independently on purpose. The
script never uses Form 2 to compute Form 1 or the reverse. If they agree, it
is because the physics agrees.

### 5.4 `main()`: setting up the LINAC

```python
# TODO: Change the LINAC accelerating gradient (MeV per meter)
gradient_mev_per_m = 20.0
```

**This is the line to edit.**

```python
rest_energy_mev = m_e * c**2 / (1e6 * e)
r_e = physical_constants["classical electron radius"][0]
```

The rest energy $m_e c^2$ is converted from joules to MeV by dividing by
$10^6 e$, because 1 eV $= e$ joules. The result is about 0.511 MeV.

```python
field = gradient_mev_per_m * 1e6  # volts per meter
force = e * field                 # newtons, equal to dp/dt
```

A gradient of 20 MeV/m on a charge $e$ is a field of $2 \times 10^7$ V/m.
The force follows from $F = e\mathcal{E}$, as in section 3.1.

### 5.5 `main()`: the range of energies

```python
total_energy_mev = np.logspace(np.log10(rest_energy_mev * 1.001), 4, 400)
gamma = total_energy_mev / rest_energy_mev
beta = np.sqrt(1 - 1 / gamma**2)
```

- `np.logspace` spaces 400 energies evenly on a log scale from just above
  the rest energy up to $10^4$ MeV (10 GeV). Log spacing gives equal
  resolution in every decade, from slow electrons to very fast ones.
- **Why start at 1.001 times the rest energy rather than exactly the rest
  energy?** At $\gamma = 1$ the electron is at rest, so $\beta = 0$. The loss
  fraction divides by $\beta c$, and that would be a division by zero.
- Total energy is $E = \gamma mc^2$, so $\gamma = E / mc^2$, and $\beta$
  follows from the definition of $\gamma$.

### 5.6 `main()`: both forms side by side

```python
accel = force / (gamma**3 * m_e)
dp_dt = np.full_like(gamma, force)

p_form1 = power_from_acceleration(accel, gamma)
p_form2 = power_from_force(dp_dt)
```

This is the core of the script.

- `accel` applies $a = F/(\gamma^3 m)$ from section 2.3. It is the actual
  acceleration the electron experiences at each energy.
- `dp_dt` is the constant force, repeated into an array the same shape as
  `gamma` so both forms can be compared element by element.
- Form 1 receives a **shrinking** acceleration together with a **growing**
  $\gamma^6$. Form 2 receives a **constant** force. They should give the same
  answer.

### 5.7 `main()`: the loss fraction

```python
energy_gain_rate = force * beta * c
loss_fraction = p_form1 / energy_gain_rate
loss_estimate = (2 / 3) * r_e * gradient_mev_per_m / rest_energy_mev
```

- `energy_gain_rate` is $dE/dt = Fv$ from section 3.3.
- `loss_fraction` is the exact ratio at every energy.
- `loss_estimate` is the closed-form $\beta \to 1$ result. The MeV units
  cancel between the gradient and the rest energy, which leaves $r_e$ in
  meters times a "per meter" gradient, a pure number.

### 5.8 `main()`: the printed table

For six sample energies the script finds the nearest grid point with
`np.argmin(np.abs(...))` and prints one row. A run at 20 MeV/m produces:

```text
   E (MeV)      gamma    a (m/s^2)   P form 1 (W)   P form 2 (W)  P_rad/(dE/dt)
     0.593      1.161   2.2458e+18   7.063364e-17   7.063364e-17     1.4460e-13
     0.998      1.953   4.7186e+17   7.063364e-17   7.063364e-17     8.5592e-14
     9.987     19.544   4.7124e+14   7.063364e-17   7.063364e-17     7.3624e-14
    99.912    195.522   4.7061e+11   7.063364e-17   7.063364e-17     7.3529e-14
   999.558   1956.087   4.6999e+08   7.063364e-17   7.063364e-17     7.3528e-14
 10000.000  19569.512   4.6937e+05   7.063364e-17   7.063364e-17     7.3528e-14

Largest relative difference between forms: 6.98e-16
Ultra-relativistic loss estimate (2/3)(r_e/mc^2)(dE/dx) = 7.3528e-14
```

How to read it:

- **Acceleration column.** It drops by a factor of 1,000 for every factor of
  10 in $\gamma$. That is the $\gamma^{-3}$ law.
- **Two power columns.** They are identical and constant in every digit
  shown.
- **Largest relative difference, $7 \times 10^{-16}$.** This is a few times
  the double-precision machine epsilon of $2.2\times10^{-16}$. The forms
  differ only by floating-point rounding.
- **Loss column.** It settles onto the analytic estimate as $\beta \to 1$
  and is larger at low energy because of the $1/\beta$ factor.

### 5.9 `main()`: the plots

**Left panel: the cancellation.**

```python
ax1.loglog(gamma, accel**2 / accel[0] ** 2, ...)
ax1.loglog(gamma, gamma**6 / gamma[0] ** 6, ...)
ax1.loglog(gamma, (accel**2 * gamma**6) / (accel[0] ** 2 * gamma[0] ** 6), "k--", ...)
```

Each curve is divided by its value at the lowest energy, so all three start
at 1 and can share one axis. On log-log axes a power law becomes a straight
line with slope equal to the exponent:

- $a^2$ falls with slope $-6$.
- $\gamma^6$ rises with slope $+6$.
- Their product is a flat line at 1.

The two lines span more than 50 orders of magnitude together, yet the
product stays exactly flat.

**Right panel: the agreement.**

```python
ax2.semilogx(total_energy_mev, p_form1, lw=5, alpha=0.5, ...)
ax2.semilogx(total_energy_mev, p_form2, "k--", ...)
```

Form 1 is drawn as a thick, semi-transparent band, and Form 2 as a thin
dashed black line on top of it. If they disagreed anywhere you would see the
dashed line leave the band. The y-axis starts at zero, so the flatness is
honest and not an artifact of a zoomed-in axis. The loss fraction at the
highest energy is printed in the corner.

---

## 6. Things to Try

1. **Raise the gradient.** Try 100 MeV/m, which is near the limit of
   today's best RF structures, and then 1 GeV/m, which plasma wakefield
   accelerators have demonstrated. The loss fraction scales linearly with the
   gradient. How large would the gradient need to be for a loss of 1%?
2. **Perpendicular force.** Write a third function for
   $P_\perp = q^2\gamma^2 (dp/dt)^2 / (6\pi\varepsilon_0 m^2 c^3)$ and add it
   to the right-hand plot. Watch it climb as $\gamma^2$. You will need a log
   y-axis.
3. **A proton LINAC.** Replace `m_e` with `m_p` from `scipy.constants` and
   adjust the rest energy. The radiated power falls by $(m_p/m_e)^2 \approx
   3.4 \times 10^6$. Why does the mass appear squared in the denominator?
4. **Energy radiated over a whole machine.** SLAC's linac is about 3 km
   long. Multiply the radiated power by the transit time $L/c$ to find the
   total energy radiated by one electron, and compare it with the 50 GeV it
   gains.
5. **Break the agreement on purpose.** Change `gamma**3` to `gamma` in the
   `accel` line, which is the perpendicular-force relation used in the wrong
   place. Form 1 and Form 2 now disagree by $\gamma^4$. This is a good way to
   see why the parallel and perpendicular cases must not be mixed up.

---

## 7. Summary

- Form 1, $P = q^2 a^2\gamma^6/(6\pi\varepsilon_0 c^3)$, and Form 2,
  $P = q^2 (dp/dt)^2/(6\pi\varepsilon_0 m^2c^3)$, are the **same equation**,
  linked by $dp/dt = \gamma^3 m a$ for a force along the motion.
- In a LINAC the force is constant, so the radiated power is **constant**,
  about $7\times10^{-17}$ W at 20 MeV/m, no matter how fast the electron is.
- The fraction of energy gain lost to radiation is
  $\tfrac{2}{3}(r_e/mc^2)(dE/dx)/\beta \approx 7\times10^{-14}$, which is
  negligible.
- In a ring the force is perpendicular, the power grows as $\gamma^4$ at
  fixed radius, and losses become severe. That is why the highest-energy
  electron machines are planned as straight lines.
- `linac_electron_radiation_power.py` checks all of this numerically. It
  shows the two forms agreeing to machine precision and the $\gamma^6$ being
  cancelled exactly by the shrinking acceleration.
