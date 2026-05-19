import numpy as np

from utils import *
from scipy.optimize import newton

def b(t, T, K, r, sigma, n):
    b_guess = K
    t_delta = T / n
    t_set = np.linspace(t, T, n+1)[:-1][::-1]  # Backward in time

    b_final = [K]

    for idx, i in enumerate(t_set):  # Exclude T itself

        def f(x, i=i):
            M = K - x

            n_steps = int(round((T - i) / t_delta))
            future_times = np.linspace(i, T, n_steps + 1)

            trap_sum = 0

            for k in range(len(future_times) - 1):
                t_left = future_times[k]
                t_right = future_times[k + 1]

                b_at_right = b_final[k]  # K at T is b_final[-1]
                b_at_left = b_final[k-1] if k > 0 else 10

                g_left = G(t_left, i, b_at_left, x, K, r, sigma)
                g_right = G(t_right, i, b_at_right, x, K, r, sigma)

                trap_sum += (g_left + g_right) * t_delta / 2

            g = P_e(i, x, T, K, r, sigma)

            return M - trap_sum - g

        # Solve for boundary at time i
        b_i = newton(f, b_guess, maxiter=100)

        # Append to beginning since we're going backwards
        b_final.insert(0, b_i)

        # Update guess for next iteration
        b_guess = b_i

        print(idx)

    return b_final


def P_a(t, T, s, K, r, sigma, n):
    boundary = b(t, T, K, r, sigma, n)
    trap_sum = 0
    t_delta = T / n

    # Create time points matching how b() creates them
    time_points = np.linspace(t, T, n+1)

    for i in range(len(time_points)-1):
        t_left = time_points[i]
        t_right = time_points[i + 1]

        b_at_left = boundary[i]
        b_at_right = boundary[i+1]

        g_left = G(t_left, t, b_at_left, s, K, r, sigma)
        g_right = G(t_right, t, b_at_right, s, K, r, sigma)

        trap_sum += (g_left + g_right) * t_delta / 2

    price = P_e(t, s, T, K, r, sigma) + trap_sum  # Fixed argument order

    return price

print(P_a(0,1,40,40,0.06,0.2,100))