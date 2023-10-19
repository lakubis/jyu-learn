# PyCon 2019 Baltimore
# Stefan Karpinski
# around 3:30 in 
# https://youtu.be/kc9HwsxE1OY
# see https://juliaphysics.github.io/Measurements.jl/stable/
# 
using DifferentialEquations
using Measurements
# in julia console:
# include("julia-errorbars.jl")
#
using Plots

g = 10.79 ± 0.02 # Gravitational constant
L = 1.00 ± 0.01 # Length of pendulum


# initial speed and angle
u0 = [0 ± 0, π/60 ± 0.01]
tspan = (0.0, 2π)

# Define the problem
# Î¸''(t) + (g/L)*Î¸(t) = 0
# u[2] := Î¸'(t) => u[2]' = Î¸''(t)
# insert
# u[2]' + g/L*Î¸(t) = 0 <=> u[2]' =  -(g/L)*Î¸ := du[2]
function pendulum(du, u, p, t)
    θ  = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -(g/L)*θ
end

# Pass to solvers
prop = ODEProblem(pendulum, u0 , tspan)
sol = solve(prop, Tsit5(), reltol = 1e-6) ;

# Analytic solution
u = u0[2] .* cos.(sqrt(g/L) .* sol.t) ;

f = plot(sol.t, u, label="analytic")
f = plot!(sol.t, sol(sol.t)[2,:] .+ .1, label="numerical (shifted up +.1) ")


# give the plot a chance to show when run from console
display(f)
readline()