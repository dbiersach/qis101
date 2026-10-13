# How Much of a LINAC Electron's Energy Becomes Bremsstrahlung?

Companion notes for `linac_electron_bremsstrahlung_power.py`.

This document explains the physics behind the script and then walks through
the code one piece at a time. Open it in VS Code with **Markdown: Open
Preview** (`Ctrl+Shift+V`) so the equations render.

This is the fourth of four documents on LINAC electrons and radiation.
`linac_electron_radiation_power.md` shows that an electron radiates almost
nothing while it is being accelerated. `linac_electron_bremsstrahlung_direction.md`
shows where the radiation goes once the beam hits a target. This document
asks how *much* of the beam energy comes out of the target as X-rays, and how
much ends up as heat.

---

## 1. The Question

A LINAC delivers electrons to a tungsten target. Some of their energy leaves
as bremsstrahlung photons. The rest heats the tungsten. What is the split, and
how does it depend on the beam energy?

The short answers, for tungsten:

| Beam | Kinetic energy | Share radiated | Share left as heat |
| --- | --- | --- | --- |
| Dental X-ray tube | 100 keV | 1.0% | 99% |
| Medical LINAC | 6 MeV | 22% | 78% |
| Medical LINAC | 10 MeV | 30% | 70% |
| Research LINAC | 100 MeV | 75% | 25% |
| Research LINAC | 1 GeV | 95% | 5% |

Two things stand out. At the energies of an X-ray tube the target is almost
purely a heater, which is why tube anodes rotate and why their output is
limited by heat rather than by physics. And even at 10 MeV most of the beam
power still goes into heating the target, which is why LINAC targets are
water-cooled.

The contrast with the LINAC itself is enormous. The electron radiates a
fraction $7 \times 10^{-14}$ of its energy gain while it is being accelerated.
It radiates 30 percent of everything it has when it hits the target. The
ratio is $4 \times 10^{12}$.

---

## 2. Two Ways to Lose Energy

An electron moving through matter loses kinetic energy continuously. The
**stopping power** is the loss per unit path length,

$$
S = -\frac{dT}{dx} .
$$

Dividing by the density gives the **mass stopping power** $S/\rho$, in units
of MeV cm$^2$/g. That is the quantity tabulated and plotted here. It is used
because it depends only weakly on the state of the material: a gram of
tungsten vapor and a gram of tungsten metal stop electrons almost equally
well, so the mass stopping power is a property of the element.

Two processes contribute, and they behave very differently.

| | Collision loss, $S_{col}$ | Radiative loss, $S_{rad}$ |
| --- | --- | --- |
| Partner | Atomic electrons | Atomic nuclei |
| What happens | Ionization and excitation | Bremsstrahlung photon emitted |
| Energy ends up as | Heat | X-rays |
| Scales with | $Z/A \approx 1/2$, almost the same for every material | $Z^2/A$, much larger for heavy elements |
| Energy dependence | Falls as $1/\beta^2$, then nearly flat | Rises in proportion to $E$ |

Because one is flat and the other rises, there is always an energy above
which radiation wins. Because one goes as $Z$ and the other as $Z^2$, that
energy is lower for heavy elements. Both facts are visible in the plot.

---

## 3. Collision Stopping Power: The Bethe Formula

### 3.1 The formula

For electrons the Bethe formula, in the form used by ICRU Report 37 and by
NIST's ESTAR program, is

$$
\frac{S_{col}}{\rho}
= \frac{K}{2}\,\frac{Z}{A}\,\frac{1}{\beta^2}
\left[\ln\frac{\tau^2(\tau+2)}{2\,(I/m_e c^2)^2} + F(\tau) - \delta\right],
\qquad
\tau = \frac{T}{m_e c^2},
$$

with

$$
F(\tau) = 1 - \beta^2 + \frac{\tau^2/8 - (2\tau+1)\ln 2}{(\tau+1)^2} .
$$

### 3.2 What each piece does

- $K = 4\pi N_A r_e^2 m_e c^2 = 0.3071$ MeV cm$^2$/mol collects the
  constants. It is the same $K$ that appears in the Bethe formula for heavy
  charged particles. The factor $1/2$ is specific to electrons.
- $Z/A$ is the number of atomic electrons per gram, up to $N_A$. It is close
  to $1/2$ for light elements and drops to $0.40$ for tungsten, which is why
  collision loss hardly depends on the material.
- $1/\beta^2$ dominates at low energy. A slow electron spends longer near
  each atom it passes and transfers more momentum, so the loss rises steeply
  as the electron slows. This is the left-hand rise in the plot.
- The logarithm compares the largest energy the electron can transfer, which
  grows with $T$, with the **mean excitation energy** $I$ of the atom.
  $I$ is an average over all the ways the atom can absorb energy. It is
  727 eV for tungsten, 322 eV for copper, 75 eV for water. A larger $I$ makes
  the atom harder to excite and lowers the stopping power.
- $F(\tau)$ gathers the relativistic corrections, including the fact that
  the incoming electron and the struck electron are identical particles.
- $\delta$ is the **density effect**, described next.

The $1/\beta^2$ fall and the logarithmic rise cross near $\tau \approx 3$,
giving a broad minimum. For tungsten the minimum sits at about 1 MeV, where
$S_{col}/\rho = 1.016$ MeV cm$^2$/g. Electrons near that energy are called
**minimum ionizing**.

### 3.3 The density effect

A fast charge polarizes the atoms it passes. In a dense medium those induced
dipoles partly cancel the electron's field at large distances, so distant
atoms receive less energy than the bare formula predicts. The reduction
$\delta$ grows as $\ln(\beta\gamma)$ at high energy and flattens the
relativistic rise.

Sternheimer gave a parameterization in terms of $x = \log_{10}(\beta\gamma)$
with five material constants $\bar{C}$, $x_0$, $x_1$, $a$, $k$ and a sixth,
$\delta_0$, for conductors:

$$
\delta =
\begin{cases}
2\ln(10)\,x - \bar{C} & x \ge x_1 \\
2\ln(10)\,x - \bar{C} + a\,(x_1 - x)^k & x_0 \le x < x_1 \\
\delta_0\,10^{\,2(x - x_0)} & x < x_0
\end{cases}
$$

The script takes the tungsten constants from the Particle Data Group tables.
At 100 MeV the correction is $\delta = 5.4$, which lowers the bracket from
about 20 to about 15. Without it the Bethe curve would overshoot the data by
a third at high energy.

### 3.4 A check

At $T = 1$ MeV in tungsten, $\tau = 1.957$, $\beta^2 = 0.8856$, and the
bracket is $15.13 - 0.22 - 0.35 = 14.56$. The prefactor is
$0.1535 \times 0.4025 / 0.8856 = 0.0698$, so $S_{col}/\rho = 1.016$. ESTAR
gives 1.016. The script's table shows the same agreement at every energy
from 10 keV to 1 GeV, to three figures.

---

## 4. Radiative Stopping Power

### 4.1 Bethe and Heitler

An electron passing a nucleus of charge $Ze$ is deflected by the Coulomb
field, and the deflection radiates. The quantum calculation (Bethe and
Heitler, 1934) gives a cross section proportional to $Z^2 \alpha r_e^2$,
where $\alpha$ is the fine-structure constant and $r_e$ the classical
electron radius. The $Z^2$ is the signature of a process driven by the
nuclear charge, and it is why tungsten ($Z = 74$) is the target of choice.

At high energy the photon spectrum stretches from zero up to the electron's
full energy, roughly flat in energy per unit photon energy, and the energy
lost per unit path is **proportional to the electron's energy**:

$$
\frac{S_{rad}}{\rho} = \frac{E}{X_0},
\qquad E = T + m_e c^2 .
$$

### 4.2 The radiation length

$X_0$ is the **radiation length**: the thickness of material over which a
high-energy electron loses all but $1/e$ of its energy to bremsstrahlung.
Tsai's formula gives it from $Z$ and $A$ alone:

$$
\frac{1}{X_0} = 4\alpha r_e^2 \frac{N_A}{A}
\Big\{ Z^2\,[L_{rad} - f(Z)] + Z\,L'_{rad} \Big\},
\qquad
L_{rad} = \ln\!\big(184.15\,Z^{-1/3}\big),\quad
L'_{rad} = \ln\!\big(1194\,Z^{-2/3}\big).
$$

The $Z^2$ term is the nuclear field, screened by the atomic electrons, which
is where the odd numbers 184.15 and the $Z^{-1/3}$ come from. The $Z$ term is
bremsstrahlung in the field of the atomic electrons themselves. $f(Z)$ is
the **Coulomb correction**,

$$
f(Z) = a^2\left[\frac{1}{1+a^2} + 0.20206 - 0.0369\,a^2 + 0.0083\,a^4 - 0.002\,a^6\right],
\qquad a = \alpha Z,
$$

which accounts for the fact that the electron's wave is distorted by the
nuclear field. For tungsten $a = 0.54$ and $f = 0.28$, a 7 percent effect.

The script evaluates this and gets $X_0 = 6.763$ g/cm$^2$ for tungsten. The
Particle Data Group lists 6.76. At a density of 19.3 g/cm$^3$ that is 3.5 mm.
For comparison, copper has $X_0 = 12.86$ g/cm$^2$ and water 36.08.

### 4.3 Where $E/X_0$ fails

The formula $S_{rad} = E/X_0$ is the **complete screening** limit. It assumes
the electron is energetic enough that the relevant impact parameters are
larger than the atom, so the nuclear charge is fully screened. That holds
above a few hundred MeV. Below that, screening is incomplete, the photon
spectrum is no longer flat, and the Born approximation behind the formula
degrades for a heavy nucleus. The script prints both numbers so the
discrepancy can be seen:

| $T$ | ESTAR $S_{rad}/\rho$ | $E/X_0$ | Ratio |
| --- | --- | --- | --- |
| 10 keV | 0.0198 | 0.0770 | 3.9 |
| 100 keV | 0.0408 | 0.0903 | 2.2 |
| 1 MeV | 0.116 | 0.223 | 1.9 |
| 10 MeV | 1.132 | 1.554 | 1.37 |
| 100 MeV | 13.55 | 14.86 | 1.10 |
| 1 GeV | 145.2 | 147.9 | 1.02 |

This is why the script does not compute the radiation yield from the formula.
Medical LINACs sit at 6 to 25 MeV, right where the formula is 30 to 40
percent high. Instead it reads the tabulated stopping powers, which come
from the full Seltzer and Berger calculation, and uses the formula only as
the high-energy reference line in the plot.

### 4.4 A rule of thumb

Dividing the two stopping powers and keeping only the leading dependence
gives the often quoted estimate

$$
\frac{S_{rad}}{S_{col}} \approx \frac{Z\,T}{800\ \text{MeV}} .
$$

For tungsten at 10 MeV this predicts 0.93; the tables give 0.94. It is good
to about 20 percent above 1 MeV and should not be trusted below.

---

## 5. The Critical Energy

The energy at which radiative and collision losses are equal is called the
**critical energy** $E_c$. Above it the electron radiates faster than it
ionizes, and an electromagnetic shower can develop. The script finds the
crossing by interpolating the tables:

| Material | $S_{rad} = S_{col}$ at | PDG critical energy |
| --- | --- | --- |
| Tungsten | 10.6 MeV | 7.97 MeV |
| Copper | 24.3 MeV | 19.42 MeV |
| Water | 90.8 MeV | 78.33 MeV |

The two columns differ because there are two definitions in use. The first is
the plain crossing. The second, due to Rossi and used by the Particle Data
Group, is the energy at which the collision loss *per radiation length*
equals the electron's energy, $S_{col} X_0 = E$. The Rossi definition is
the one that makes shower theory come out cleanly, and it is always somewhat
lower. Either way the ordering is the same: heavy elements have low critical
energies because $S_{rad}$ carries the $Z^2$.

Two consequences:

- **Why the target is tungsten and the beam is 6 to 25 MeV.** A medical
  LINAC runs near tungsten's critical energy, where a thin target converts a
  usable fraction of the beam and still stops it in a few millimeters.
- **Why the patient does not radiate.** Tissue is close to water. At 6 MeV
  the ratio in water is 0.05, and the yield is 2 percent, compared with 22
  percent in tungsten. The X-rays are made in the target, not in the
  patient.

---

## 6. The Radiation Yield

### 6.1 The integral

The stopping-power ratio at the beam energy does not by itself say how much
of the beam energy becomes X-rays, because the electron does not stay at
that energy. It slows down, and as it slows the radiative share drops. The
**radiation yield** $Y(T_0)$ averages the share over the whole slowing-down
history.

In a short stretch of path the electron loses $dT$. Of that, the fraction

$$
\frac{S_{rad}}{S_{col} + S_{rad}}
$$

goes into photons. Summing from the initial energy $T_0$ down to rest and
dividing by $T_0$,

$$
\boxed{\;
Y(T_0) = \frac{1}{T_0}\int_0^{T_0} \frac{S_{rad}(T)}{S_{col}(T) + S_{rad}(T)}\,dT
\;}
$$

Because the integrand is smaller at every $T$ below $T_0$ than it is at
$T_0$, the yield is always less than the share at the beam energy. For
tungsten at 10 MeV the share is $1.132/(1.203 + 1.132) = 0.485$, but the
yield is 0.30.

### 6.2 What the integral assumes

- **Thick target.** The electron stops completely in the target.
- **Continuous slowing down.** The electron loses energy smoothly along its
  path, at the mean rate given by the stopping powers. This is the CSDA
  (continuous slowing down approximation) that ESTAR also uses.
- **Every photon counts.** The yield is the energy *emitted* as photons. In
  a real target some of those photons are absorbed before they get out, and
  some electrons scatter back out of the surface before they stop. The
  energy that actually leaves the target as X-rays is lower, especially for
  thick, high-$Z$ targets at low energy.

### 6.3 Starting the integral

The tables start at 10 keV, not at zero. Below that energy $S_{col}$ grows as
$1/\beta^2 \propto 1/T$ while $S_{rad}$ is nearly flat, so the share falls
roughly in proportion to $T$. The script treats the missing stretch as a
triangle of area $\tfrac{1}{2}\,\text{share}(T_1)\,T_1$, where $T_1$ is the
first tabulated energy. That is exact if the share is linear in $T$ and a
few percent off if it is not, which is why the yield at 10 keV differs from
NIST's by 2 percent for tungsten and 8 percent for water. From 100 keV
upward the agreement is within 0.2 percent, and from 1 MeV it is within
0.05 percent.

---

## 7. The Numbers in Context

**X-ray tubes.** A tungsten anode at 100 kV converts 1.0 percent of the beam
energy into photons. The classic estimate due to Kramers,
$\eta \approx 10^{-9}\,Z\,V$ with $V$ in volts, gives 0.7 percent. A 100 kV,
1 kW tube therefore delivers about 10 W of X-rays and 990 W of heat into a
spot a millimeter across. The anode rotates so that each patch of the track
has time to cool.

**Medical LINACs.** At 6 MeV the tungsten yield is 22 percent; at 10 MeV it
is 30 percent. The beam is typically a few hundred watts, so the target
absorbs on the order of 100 to 200 W and is cooled by water flowing through
the target holder. Because the photons are strongly forward-peaked (the
subject of the direction script), a flattening filter below the target
spreads the useful beam across the treatment field.

**Research machines.** At 1 GeV the yield is 95 percent. Almost nothing is
left for heat, and a thin radiator rather than a thick target is used when a
photon beam is wanted, because the electron can be reused or dumped
elsewhere.

**The LINAC itself.** `linac_electron_radiation_power.py` found that at a
gradient of 20 MeV/m an electron radiates the fraction

$$
\frac{P_{rad}}{dE/dt} = \frac{2}{3}\,\frac{r_e}{m_e c^2}\,\frac{dE}{dx}
= 7.35 \times 10^{-14}
$$

of its energy gain. The same electron, stopping in tungsten at 10 MeV,
radiates 0.30 of everything it has. The ratio is $4 \times 10^{12}$. The
reason is the field strength. The accelerating field is $2 \times 10^7$ V/m.
The field of a tungsten nucleus at a distance of $10^{-12}$ m, a hundredth
of the atomic radius, is $Ze/(4\pi\varepsilon_0 r^2) \approx 10^{17}$ V/m,
ten billion times larger, and the radiated power goes as the square of the
force. The target wins by a factor of $10^{20}$ per unit time and loses most
of it back because the electron spends so little time that close to any
nucleus.

---

## 8. The Data

The stopping powers come from **ESTAR**, the electron stopping-power and
range program of the National Institute of Standards and Technology:

```text
https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html
```

ESTAR computes collision stopping powers from the Bethe formula of Section 3
with the mean excitation energies of ICRU Report 37, and radiative stopping
powers from the tabulated cross sections of Seltzer and Berger, which cover
the low-energy region where the simple formula fails. It also integrates
the radiation yield in the way described in Section 6, which is why its
yield column is a good check on the script.

Three tables were saved beside the script as comma-separated files:

- `estar_tungsten.csv`
- `estar_copper.csv`
- `estar_water.csv`

Each holds ESTAR's default grid of 81 kinetic energies from 10 keV to 1 GeV
with four columns: kinetic energy in MeV, collision stopping power, radiative
stopping power (both in MeV cm$^2$/g), and NIST's own radiation yield. Lines
beginning with `#` record the source, the density, and the mean excitation
energy. ESTAR's other columns (total stopping power, CSDA range, density
effect) are not needed here and were left out.

---

## 9. Walking Through the Code

### 9.1 Header and constants

```python
#!/usr/bin/env -S uv run
"""linac_electron_bremsstrahlung_power.py"""
```

The constants `N_A`, `alpha`, `c`, `e`, and `m_e` come from `scipy.constants`,
and the classical electron radius from its `physical_constants` dictionary.
Three module constants are built from them: `REST_ENERGY_MEV`, the electron
radius in centimeters `R_E_CM` (stopping powers are in cm$^2$/g, so every
length here is in centimeters), and the Bethe constant `K_BETHE`.

`TUNGSTEN` is a dictionary holding $Z$, $A$, $I$, and the six Sternheimer
parameters. It is the only material the script treats theoretically; the
others appear through their data files alone.

### 9.2 `read_estar(file_path)`

Reads one of the CSV files with `np.genfromtxt`. `comments="#"` skips the
header lines and `unpack=True` returns the four columns as four arrays. The
path is built by the caller from `Path(__file__).parent`, so the files are
found next to the script no matter which folder the script is launched from.

### 9.3 `density_effect(beta_gamma, material)`

Implements the three-branch Sternheimer formula of Section 3.3. The branches
are evaluated for every energy and then selected with two nested `np.where`
calls, which is the usual way to write a piecewise function over a NumPy
array.

### 9.4 `bethe_collision(kinetic_mev, material)`

The Bethe formula, term by term, in the order of Section 3.2. The mean
excitation energy is converted from eV to a fraction of the rest energy
before it enters the logarithm.

### 9.5 `radiation_length(z, a)`

Tsai's formula of Section 4.2, returning $X_0$ in g/cm$^2$. The polynomial
in `coulomb` is $f(Z)$.

### 9.6 `radiation_yield(kinetic, s_col, s_rad)`

The integral of Section 6. `share` is the integrand. `cumulative_trapezoid`
from `scipy.integrate` returns the running integral at every tabulated
energy in one call, with `initial=0` so the output has the same length as
the input. The triangle of Section 6.3 is added as a constant, and dividing
by `kinetic` turns the integral into a yield.

### 9.7 `crossover_energy(kinetic, s_col, s_rad)`

Finds the first tabulated energy where $S_{rad} \ge S_{col}$ and interpolates
between it and the previous one. Both stopping powers are close to straight
lines on log-log axes, so the interpolation is done in $\ln T$ and
$\ln(S_{rad}/S_{col})$.

### 9.8 `main()`: reading the tables

```python
# TODO: Change the target materials to compare. Each entry names an ESTAR
# data file that sits in the same folder as this script
materials = {
    "Tungsten": "estar_tungsten.csv",
    "Copper": "estar_copper.csv",
    "Water": "estar_water.csv",
}
```

Each file is read once and its arrays, together with the yield integrated
from them, are stored in `tables` under the material name. The full path of
every file is printed so a student can see where the data came from.

### 9.9 `main()`: the table

For tungsten the script computes $X_0$, evaluates the Bethe formula and
$E/X_0$ on a fine grid for the plot, and prints a table at seven energies:
the ESTAR collision stopping power beside the Bethe value, the ESTAR
radiative stopping power beside $E/X_0$, their ratio, and the yield from the
script's integration beside NIST's. Then, for every material, it prints the
crossing energy and the largest relative mismatch between the two yields.
Finally it recomputes the in-LINAC loss fraction from the formula in
Section 7 and compares it with the 10 MeV tungsten yield.

### 9.10 `main()`: the plots

Three panels side by side.

- **Left.** Both tungsten stopping powers on log-log axes, with the Bethe
  curve dashed on top of $S_{col}$ and $E/X_0$ dotted beside $S_{rad}$. A
  vertical line marks the crossing.
- **Middle.** The ratio $S_{rad}/S_{col}$ for every material, with a
  horizontal line at 1 and a dot where each curve crosses it. The crossing
  energy is written into each legend entry.
- **Right.** The radiation yield. Solid lines are the script's integration;
  open circles every sixth tabulated energy are NIST's yield column, so
  agreement shows as circles sitting on the lines. Dotted vertical lines
  mark 100 keV and 6 MeV, labeled with the tungsten yield at each.

---

## 10. Reading the Plot

**Left panel.** The blue collision curve falls as $1/\beta^2$, bottoms out
near 1 MeV, and creeps up again. The dashed Bethe line sits on top of it
everywhere, which says the formula of Section 3 is the whole story for
collision loss. The red radiative curve climbs across four decades. The
dotted $E/X_0$ line meets it only at the top right, and is well above it
through the MeV range, which is the failure described in Section 4.3. The
two solid curves cross at 10.6 MeV.

**Middle panel.** Three nearly parallel lines, one per material, each
sloping up at roughly one decade per decade: the ratio is proportional to
$T$. Tungsten sits highest because of $Z^2$. Each line crosses 1 at that
material's critical energy, from 11 MeV for tungsten to 91 MeV for water.

**Right panel.** The yield is an S-curve on a log energy axis. It is below
1 percent through the X-ray tube range, passes 22 percent at the 6 MeV of a
medical LINAC, and saturates toward 1 only around a GeV. Water's curve is
the same shape shifted a decade to the right. The open circles from NIST sit
on the lines.

---

## 11. Things to Try

- **Add a material.** Go to the ESTAR page, pick lead or aluminum, submit
  with the default grid, and save the kinetic energy, collision, radiative,
  and radiation yield columns as a four-column CSV beside the script. Add it
  to `materials` and rerun. Lead ($Z = 82$) should cross slightly below
  tungsten; aluminum ($Z = 13$) well above copper.
- **Test the rule of thumb.** Plot $Z\,T/800$ MeV on the middle panel for
  tungsten and see where it departs from the tabulated ratio.
- **Compute the range.** The CSDA range is
  $R = \int_0^{T_0} dT / (S_{col} + S_{rad})$. Integrate it the same way as
  the yield and compare with ESTAR's range column, which is on the web page
  but not in the CSV files. For 10 MeV electrons in tungsten it is
  6.2 g/cm$^2$, or 3.2 mm.
- **Turn off the density effect.** Return zero from `density_effect` and
  watch the Bethe curve rise above the data at high energy.
- **Change the gradient** in the final comparison. Even a 100 MeV/m
  gradient, beyond anything built, leaves the in-LINAC loss at
  $4 \times 10^{-13}$.

---

## 12. Summary

- An electron in matter loses energy by collisions, which heat the material,
  and by bremsstrahlung, which makes X-rays. Collision loss goes as $Z/A$
  and is nearly flat with energy; radiative loss goes as $Z^2/A$ and rises in
  proportion to $E$.
- The Bethe formula with the density effect reproduces the tabulated
  collision stopping power of tungsten to three figures from 10 keV to
  1 GeV.
- The radiation length from Tsai's formula is 6.76 g/cm$^2$ for tungsten,
  and $S_{rad} = E/X_0$ is accurate above a few hundred MeV but 30 to 40
  percent high in the medical range, so the yield is integrated from the
  NIST tables instead.
- Radiative and collision losses are equal at 10.6 MeV in tungsten, 24 MeV
  in copper, and 91 MeV in water.
- The radiation yield of a thick tungsten target is 1 percent at 100 keV,
  22 percent at 6 MeV, 30 percent at 10 MeV, and 95 percent at 1 GeV. The
  rest is heat, which is why targets are cooled.
- That 30 percent is $4 \times 10^{12}$ times the fraction the same electron
  radiated while being accelerated in the LINAC.
