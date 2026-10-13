# Synchrotron Radiation: How Much It Costs

Companion notes for `synchrotron_radiation_power.py`.

This document explains the physics behind the script and then walks through
the code one piece at a time. Open it in VS Code with **Markdown: Open
Preview** (`Ctrl+Shift+V`) so the equations render.

`synchrotron_radiation_direction.md` shows where an electron on a circular
orbit radiates. This document asks how much energy that costs per turn, what
photon energies come out, and why the answer decided the shape of every
large electron machine built in the last fifty years.

---

## 1. The Question

An electron storage ring bends its beam on a circle of radius $\rho$. Every
turn, each electron radiates part of its energy and the radio-frequency
cavities must put it back. How large is the loss, and how does it depend on
the beam energy and the radius?

The short answers:

- The energy lost per turn grows as the **fourth power** of the beam energy
  and falls only as the first power of the radius: $U_0 \propto E^4/\rho$.
  For electrons, $U_0 = 88.5$ keV $\times E^4$ with $E$ in GeV and $\rho$
  in meters.
- At the end of its life LEP ran at 104.5 GeV on a 3026 m radius and lost
  3.5 GeV per turn, more than 3 percent of the beam energy on every lap.
  That is why no higher-energy electron ring was ever built in that tunnel,
  and why the proposed FCC-ee needs a 91 km one.
- Protons escape because the loss also goes as the inverse fourth power of
  the mass. The LHC's 7 TeV protons on a 2804 m radius lose 6.7 keV per
  turn, a factor $(m_p/m_e)^4 = 10^{13}$ less than electrons would.
- The photons come out with a spectrum that peaks near a **critical energy**
  $\varepsilon_c \propto E^3/\rho$, which for a 3 GeV ring is a few keV. That
  is why storage rings are X-ray sources.

The script draws the loss per turn and the critical energy as colored maps
over beam energy and bending radius, with real machines marked on them, and
plots the universal spectrum.

---

## 2. From the Liénard Power to the Energy per Turn

### 2.1 Perpendicular acceleration

The direction document integrates the Liénard pattern for acceleration
perpendicular to the velocity and gets

$$
P = \frac{q^2 a^2 \gamma^4}{6\pi\varepsilon_0 c^3} .
$$

On a circle of radius $\rho$ the centripetal acceleration is
$a = v^2/\rho = \beta^2 c^2/\rho$, so

$$
P = \frac{q^2 c\,\beta^4\gamma^4}{6\pi\varepsilon_0\,\rho^2} .
$$

### 2.2 One revolution

A turn takes $2\pi\rho/(\beta c)$. Multiplying,

$$
\boxed{\;
U_0 = \frac{q^2\,\beta^3\gamma^4}{3\varepsilon_0\,\rho}
\;}
$$

This is the energy radiated per particle per turn. Nothing else about the
machine enters: not the circumference, not the number of magnets, only the
radius of the arcs where the beam is bent.

### 2.3 The practical form

Write $\gamma = E/(mc^2)$ and use the classical radius
$r_e = q^2/(4\pi\varepsilon_0 mc^2)$. For $\beta \to 1$,

$$
U_0 = \frac{4\pi}{3}\,r_e\,mc^2\,\gamma^4\,\frac{1}{\rho}
= \frac{4\pi}{3}\,\frac{r_e}{(mc^2)^3}\,\frac{E^4}{\rho}
\equiv C_\gamma\,\frac{E^4}{\rho} .
$$

The script evaluates $C_\gamma$ from the constants and prints
$8.846 \times 10^{-5}$ m/GeV$^3$, which is the number in every accelerator
handbook. In convenient units,

$$
U_0\ [\text{keV}] = 88.46\,\frac{E^4\ [\text{GeV}^4]}{\rho\ [\text{m}]} .
$$

Because $C_\gamma$ carries $(mc^2)^{-3}$ and $\gamma^4$ carries another
$(mc^2)^{-4}$ at fixed $E$, the loss at a given beam energy goes as
$1/m^4$. That single fact separates electron machines from proton machines.

---

## 3. The Fourth Power in Practice

The script prints a table for six machines. The numbers below come from that
output.

| Machine | $E$ (GeV) | $\rho$ (m) | $\gamma$ | $U_0$ per turn | $\varepsilon_c$ |
| --- | --- | --- | --- | --- | --- |
| NSLS-II | 3.0 | 25 | 5 900 | 287 keV | 2.4 keV |
| APS | 7.0 | 39 | 13 700 | 5.45 MeV | 19.5 keV |
| SPring-8 | 8.0 | 39 | 15 700 | 9.23 MeV | 28.9 keV |
| FCC-ee at the Z | 45.6 | 10 760 | 89 000 | 35.5 MeV | 19.5 keV |
| LEP2 | 104.5 | 3 026 | 205 000 | 3.49 GeV | 837 keV |
| FCC-ee at $t\bar{t}$ | 182.5 | 10 760 | 357 000 | 9.12 GeV | 1.25 MeV |

Three stories are in this table.

**Light sources.** NSLS-II, APS and SPring-8 are built to radiate. Their
small radii and modest energies give losses of a fraction of a percent per
turn, easily replaced, and critical energies in the X-ray band, which is the
product they sell.

**LEP.** Going from 8 GeV to 104.5 GeV at a radius only 77 times larger
multiplied the loss by $13^4/77 \approx 370$. LEP's cavities had to supply
3.5 GeV per turn, and the published total accelerating voltage at the end of
the run was 3.63 GV. The loss was the energy limit of the machine. Going to
120 GeV would have cost $1.7\times$ more per turn in the same tunnel.

**FCC-ee.** To reach 182.5 GeV at a loss the cavities can supply, the design
makes the ring 91 km around. Even so the loss is 9 GeV per turn, 5 percent of
the beam energy. The design fixes the radiated power at 50 MW per beam and
chooses the beam current to match, which is why the two FCC-ee rows in the
script's printout both show 49 MW at very different currents.

**The LHC.** The script also prints the proton comparison. On the LHC's
radius a 7 TeV proton loses 6.7 keV per turn; an electron at the same energy
and radius would lose $7.6 \times 10^7$ GeV, which is to say it could not
exist. The ratio is $(m_p/m_e)^4 = 1.1 \times 10^{13}$ exactly.

---

## 4. Radiated Power at the Design Current

A beam of current $I$ carries $I/q$ particles past any point per second, so
the power radiated by the whole beam is

$$
P_{\text{beam}} = U_0\,\frac{I}{q} .
$$

With $U_0$ in eV this is simply $U_0 \times I$ in watts. The script's table
gives 0.14 MW for NSLS-II at 500 mA, 0.9 MW for SPring-8, 21 MW for LEP2 at
6 mA, and 49 MW for each FCC-ee operating point. All of that power leaves the
beam as X-rays and gamma rays, strikes the vacuum chamber walls, and has to
be carried away by cooling water. It is the single largest item in the
electricity bill of a large electron ring.

---

## 5. The Spectrum and the Critical Energy

### 5.1 The flash

The direction document shows that a fixed observer in the orbit plane sees
the headlight only as it sweeps past, for a time of order $\rho/(\gamma^3 c)$.
A pulse that short contains frequencies up to about $\gamma^3 c/\rho$. The
exact calculation defines the **critical frequency**

$$
\omega_c = \frac{3}{2}\,\frac{\gamma^3 c}{\rho},
\qquad
\varepsilon_c = \hbar\omega_c = \frac{3}{2}\,\frac{\hbar c\,\gamma^3}{\rho} .
$$

In practical units for electrons,

$$
\varepsilon_c\ [\text{keV}] = 2.218\,\frac{E^3\ [\text{GeV}^3]}{\rho\ [\text{m}]} .
$$

Check: NSLS-II, $2.218 \times 27 / 25 = 2.4$ keV. The critical energy is
defined so that **half of the radiated power is above it and half below**.
The script integrates the spectrum and prints the fraction below
$\varepsilon_c$: 0.4999.

### 5.2 The universal spectrum

The power radiated per unit photon energy, for any electron on any circle,
has one shape when the photon energy is measured in units of $\varepsilon_c$.
With $x = \varepsilon/\varepsilon_c$,

$$
S(x) = \frac{9\sqrt{3}}{8\pi}\,x\int_x^\infty K_{5/3}(t)\,dt,
\qquad \int_0^\infty S(x)\,dx = 1,
$$

where $K_{5/3}$ is a modified Bessel function. Its properties:

- For $x \ll 1$ it rises as $x^{1/3}$, a slow, broad low-energy tail.
- It peaks at $x = 0.29$, below the critical energy.
- For $x \gg 1$ it dies as $e^{-x}$. There is essentially no power beyond
  $10\,\varepsilon_c$.

The spectrum stretches over several decades, from the infrared to beyond
$\varepsilon_c$, which is why a single storage ring can serve beamlines
working at very different photon energies.

---

## 6. Compared with the LINAC

`linac_electron_radiation_power.md` found that an electron being pushed
along a straight line radiates a fraction $7 \times 10^{-14}$ of the energy
it gains. An electron in LEP radiated 3 percent of its energy *per turn*.
The contrast has two sources.

- **$\gamma^4$ against a cancelling $\gamma^6$.** In the LINAC the force is
  fixed and the acceleration falls as $1/\gamma^3$, so $a^2\gamma^6$ is
  constant. In the ring the magnetic force $qvB$ is also fixed, but the
  acceleration it produces falls only as $1/\gamma$, because the transverse
  mass is $\gamma m$ rather than $\gamma^3 m$. Then $a^2\gamma^4$ grows as
  $\gamma^2$. At the same force the ring radiates $\gamma^2$ times more,
  which at LEP was $4 \times 10^{10}$.
- **A much larger force.** A LINAC gradient of 20 MV/m is a force of
  $3 \times 10^{-12}$ N. A bending field of 0.1 T on a 104.5 GeV electron
  is $qvB = 5 \times 10^{-12}$ N, comparable, but the ring applies it over
  the full 19 km of arcs on every turn, and the beam makes 11 000 turns a
  second.

Both facts point the same way. Linear machines radiate nothing worth
mentioning. Circular electron machines are limited by radiation, and above
LEP's energy the only options are a very large ring or a linear collider.

---

## 7. Walking Through the Code

### 7.1 Header and constants

```python
#!/usr/bin/env -S uv run
"""synchrotron_radiation_power.py"""
```

From `scipy.constants` come $c$, $e$, $\varepsilon_0$, $\hbar$, $m_e$, and
the classical electron radius. `kv` from `scipy.special` is the modified
Bessel function $K_\nu$, `quad` integrates its tail, and
`cumulative_trapezoid` builds the running integral of the spectrum.
`ELECTRON_MASS_MEV` and `R_E` are module constants used throughout.

### 7.2 `lorentz_factors(energy_gev, mass_mev)`

Returns $(\gamma, \beta)$ for a particle of any rest energy. Every formula
takes the mass as an argument so that the whole script can be switched to
protons or muons by changing one number.

### 7.3 `energy_loss_per_turn(energy_gev, radius_m, mass_mev)`

The boxed formula of Section 2.2, evaluated in SI units and divided by $e$
to give electron-volts. It accepts arrays, which is how the map is made.

### 7.4 `critical_photon_energy(energy_gev, radius_m, mass_mev)`

$\varepsilon_c = \tfrac{3}{2}\hbar c\gamma^3/\rho$, again in eV.

### 7.5 `synchrotron_spectrum(x)`

Evaluates $\int_x^\infty K_{5/3}(t)\,dt$ with `quad` at each requested $x$
and multiplies by $\tfrac{9\sqrt{3}}{8\pi}x$. The integral is well behaved
because $K_{5/3}$ decays exponentially, and `quad` handles the infinite upper
limit directly.

### 7.6 `format_ev(value_ev)`

Picks keV, MeV or GeV so that the table reads naturally.

### 7.7 `main()`: the settings

```python
# TODO: Change the particle. Electrons are 0.511 MeV, protons 938.27 MeV
particle_mass_mev = ELECTRON_MASS_MEV

# TODO: Change the machines drawn on the maps: (beam energy GeV, bending radius m, beam current A)
machines = { ... }
```

Each machine is a beam energy, a bending radius, and a beam current. The
radius is the radius of the dipole magnets, not the radius of the tunnel,
which is larger because the arcs are joined by straight sections. A second
dictionary places each machine's label so that neighbors on the map do not
overlap.

### 7.8 `main()`: the printout

The script prints $C_\gamma$ and the practical formula, then one row per
machine with $\gamma$, $U_0$, $\varepsilon_c$, the current, and the radiated
power in megawatts. It follows with the LHC comparison of Section 3 and the
three checks on the spectrum: that it integrates to 1, that half the power
is below $\varepsilon_c$, and where it peaks.

### 7.9 `main()`: the maps

A 400 by 400 grid covers beam energies from 10 MeV to 1 TeV and bending radii
from 1 m to 100 km, both logarithmic. `energy_loss_per_turn` and
`critical_photon_energy` are evaluated on the whole grid at once. Each map is
drawn with `pcolormesh` on log-log axes and a `LogNorm` color scale, with
white contour lines at round values (1 keV, 1 MeV, 1 GeV for the loss; 1 eV,
1 keV, 100 keV for the critical energy) labeled by `clabel`. The machines are
dots with offset labels. On log-log axes a power law $E^4/\rho$ is a straight
contour of slope 4, and $E^3/\rho$ a straight contour of slope 3, which is
visible in the plot.

### 7.10 `main()`: the spectrum

The third panel plots $S(x)$ on log-log axes from $x = 10^{-3}$ to 10, with
the region above $x = 1$ shaded to show that it holds half the power.

---

## 8. Reading the Plot

**Left panel.** Brightness is the energy lost per turn. The contours are
straight lines of slope 4 on these axes: multiply the energy by 10 and the
radius must grow by $10^4$ to keep the loss the same. The light sources sit
low on the map, at small radius and modest energy, just above the 100 keV
level. LEP2 sits on the 1 GeV contour. The two FCC-ee points are at the top,
where the radius is as large as a tunnel can practically be.

**Middle panel.** Brightness is the critical photon energy, with contours of
slope 3. The light sources sit in the keV band, which is what makes them
X-ray sources. LEP and FCC-ee sit in the hard X-ray and gamma region, which
was a nuisance for them rather than a product.

**Right panel.** The universal spectrum. The broad $x^{1/3}$ rise on the
left, the peak just below $x = 1$, and the exponential cliff on the right.
The shaded half above $\varepsilon_c$ has the same area as the unshaded half
below it, on a linear scale.

---

## 9. Things to Try

- **Protons.** Set `particle_mass_mev = 938.27`. Every contour on both maps
  moves by $(1836)^4$ in loss and $(1836)^3$ in critical energy. Add the LHC
  as a machine at (7000 GeV, 2804 m, 0.58 A) and read off its loss.
- **Muons.** Set the mass to 105.66 MeV. A muon collider's radiation problem
  is $(207)^4 = 1.8 \times 10^9$ times milder than an electron collider's,
  which is one of the arguments for building one.
- **A hypothetical ring.** Add a machine at (500 GeV, 10 760 m) and look at
  the loss per turn: more than the beam energy. The map says directly that
  such a machine cannot be a ring.
- **Scale the LEP tunnel.** Keep 104.5 GeV and vary the radius to find how
  big the tunnel would have had to be for a 1 GeV loss per turn.
- **Change the current.** Double the NSLS-II current and watch the radiated
  power double. The loss per turn is a property of each electron; the power
  is a property of the beam.

---

## 10. Summary

- An electron on a circle of radius $\rho$ loses
  $U_0 = q^2\beta^3\gamma^4/(3\varepsilon_0\rho) = 88.5\ \text{keV} \times E^4[\text{GeV}]/\rho[\text{m}]$
  per turn.
- The fourth power of the energy is why LEP stopped at 104.5 GeV with a
  3.5 GeV loss per turn, and why FCC-ee is designed 91 km around.
- The inverse fourth power of the mass is why the LHC's protons lose only
  6.7 keV per turn on a comparable radius.
- The photons have a universal spectrum peaking near the critical energy
  $\varepsilon_c = \tfrac{3}{2}\hbar c\gamma^3/\rho = 2.2\ \text{keV} \times E^3/\rho$,
  with half the power above it and half below.
- At the same force a bent electron radiates $\gamma^2$ times more than a
  pushed one, which is the whole difference between a storage ring and a
  LINAC.
