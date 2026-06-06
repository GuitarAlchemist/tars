// Fix the integration error - compare with analytical solutions

open System

printfn "🔧 FIXING INTEGRATION ERROR"
printfn "==========================="

// Known analytical solution for flat Lambda-CDM
// For ΩM = 0.315, ΩΛ = 0.685, z = 1.0
// The integral ∫[0 to z] dz'/H(z') should give a specific value

let H0 = 67.4  // km/s/Mpc
let OmegaM = 0.315
let OmegaL = 0.685
let z_target = 1.0

// Hubble parameter function
let H_of_z z = H0 * sqrt(OmegaM * (1.0 + z)**3.0 + OmegaL)

printfn "Testing integration methods:"
printfn ""

// Method 1: Our current method (wrong)
printfn "Method 1 - Current (wrong):"
let n_steps_1 = 1000
let dz_1 = z_target / float n_steps_1
let mutable integral_1 = 0.0

for i in 0 .. n_steps_1 do
    let z_i = float i * dz_1
    let H_z = H_of_z z_i
    let weight = if i = 0 || i = n_steps_1 then 0.5 else 1.0
    integral_1 <- integral_1 + weight * dz_1 / H_z

printfn "  Integral = %.6f Mpc*s/km" integral_1
printfn "  D_L = %.1f Mpc" ((299792.458 / H0) * 2.0 * integral_1)
printfn ""

// Method 2: Higher resolution
printfn "Method 2 - Higher resolution:"
let n_steps_2 = 100000
let dz_2 = z_target / float n_steps_2
let mutable integral_2 = 0.0

for i in 0 .. n_steps_2 do
    let z_i = float i * dz_2
    let H_z = H_of_z z_i
    let weight = if i = 0 || i = n_steps_2 then 0.5 else 1.0
    integral_2 <- integral_2 + weight * dz_2 / H_z

printfn "  Integral = %.6f Mpc*s/km" integral_2
printfn "  D_L = %.1f Mpc" ((299792.458 / H0) * 2.0 * integral_2)
printfn ""

// Method 3: Simpson's rule
printfn "Method 3 - Simpson's rule:"
let n_steps_3 = 10000
let dz_3 = z_target / float n_steps_3
let mutable integral_3 = 0.0

for i in 0 .. n_steps_3 do
    let z_i = float i * dz_3
    let H_z = H_of_z z_i
    let weight = 
        if i = 0 || i = n_steps_3 then 1.0
        elif i % 2 = 1 then 4.0
        else 2.0
    integral_3 <- integral_3 + weight / H_z

integral_3 <- integral_3 * dz_3 / 3.0

printfn "  Integral = %.6f Mpc*s/km" integral_3
printfn "  D_L = %.1f Mpc" ((299792.458 / H0) * 2.0 * integral_3)
printfn ""

// Method 4: Check against known Lambda-CDM result
printfn "Method 4 - Expected result check:"
// For Lambda-CDM with these parameters, D_L(z=1) ≈ 6600 Mpc
let D_L_expected = 6600.0
let c_over_H0 = 299792.458 / H0
let expected_integral = D_L_expected / (c_over_H0 * 2.0)

printfn "  Expected D_L = %.0f Mpc" D_L_expected
printfn "  Expected integral = %.6f Mpc*s/km" expected_integral
printfn "  Our integral = %.6f Mpc*s/km" integral_2
printfn "  Ratio = %.1f (we need %.1fx larger integral)" (expected_integral / integral_2) (expected_integral / integral_2)
printfn ""

// Method 5: Check units carefully
printfn "Method 5 - Unit analysis:"
printfn "  H(z) has units: km/s/Mpc"
printfn "  1/H(z) has units: Mpc*s/km"
printfn "  ∫dz/H(z) has units: Mpc*s/km"
printfn "  c/H0 has units: (km/s)/(km/s/Mpc) = Mpc"
printfn "  D_L = (c/H0) * (1+z) * ∫dz/H(z)"
printfn "      = Mpc * dimensionless * (Mpc*s/km)"
printfn "      = Mpc² * s/km"
printfn ""
printfn "  ❌ UNIT ERROR FOUND!"
printfn "  The integral should be dimensionless, not have units Mpc*s/km"
printfn ""

// Method 6: Correct calculation
printfn "Method 6 - CORRECTED calculation:"
printfn "  The correct formula is: D_L = (c/H0) * (1+z) * ∫[0 to z] dz'/H(z')"
printfn "  where H(z') is in units of H0, not km/s/Mpc"
printfn ""

// Corrected Hubble parameter (dimensionless)
let E_of_z z = sqrt(OmegaM * (1.0 + z)**3.0 + OmegaL)

let mutable integral_correct = 0.0
let n_steps_correct = 10000
let dz_correct = z_target / float n_steps_correct

for i in 0 .. n_steps_correct do
    let z_i = float i * dz_correct
    let E_z = E_of_z z_i  // Dimensionless
    let weight = if i = 0 || i = n_steps_correct then 0.5 else 1.0
    integral_correct <- integral_correct + weight * dz_correct / E_z

printfn "  Corrected integral = %.6f (dimensionless)" integral_correct
let D_L_correct = (299792.458 / H0) * (1.0 + z_target) * integral_correct
printfn "  Corrected D_L = %.1f Mpc" D_L_correct
printfn ""

printfn "🎯 SOLUTION FOUND:"
printfn "=================="
printfn "The error was in the Hubble parameter units in the integral."
printfn "We were using H(z) in km/s/Mpc, but should use E(z) = H(z)/H0 (dimensionless)."
printfn ""
printfn "Comparison:"
printfn "  Wrong method: D_L = %.1f Mpc" ((299792.458 / H0) * 2.0 * integral_2)
printfn "  Correct method: D_L = %.1f Mpc" D_L_correct
printfn "  Expected (Lambda-CDM): ~6600 Mpc"
printfn "  Improvement factor: %.1fx" (D_L_correct / ((299792.458 / H0) * 2.0 * integral_2))
printfn ""

if D_L_correct > 3000.0 && D_L_correct < 10000.0 then
    printfn "✅ FIXED! Distance is now in the correct range."
else
    printfn "⚠️ Still not quite right, but much better."
