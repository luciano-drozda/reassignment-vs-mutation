function du3_mutation_adjoint(
  u   :: Array{Float64, 3}, 
  dxu :: Array{Float64, 3}, 
  dyu :: Array{Float64, 3}, 
  dzu :: Array{Float64, 3}
)

  for k = 1:N
    for j = 1:N
      for i = 1:N
        im2 = (i + N - 3) % N + 1
        jm2 = (j + N - 3) % N + 1
        km2 = (k + N - 3) % N + 1
        im1 = (i + N - 2) % N + 1
        jm1 = (j + N - 2) % N + 1
        km1 = (k + N - 2) % N + 1

        dxu[i, j, k] += (
          u[im2, j, k] - 4u[im1, j, k] + 3u[i, j, k]
        )
        dyu[i, j, k] += (
          u[i, jm2, k] - 4u[i, jm1, k] + 3u[i, j, k]
        )
        dzu[i, j, k] += (
          u[i, j, km2] - 4u[i, j, km1] + 3u[i, j, k]
        )
      end
    end
  end

  yb   = 1.0
  ub   = zero(u)
  dxub = 2yb .* dxu
  dyub = 2yb .* dyu
  dzub = 2yb .* dzu
  for k = N:-1:1
    for j = N:-1:1
      for i = N:-1:1
        im2 = (i + N - 3) % N + 1
        jm2 = (j + N - 3) % N + 1
        km2 = (k + N - 3) % N + 1
        im1 = (i + N - 2) % N + 1
        jm1 = (j + N - 2) % N + 1
        km1 = (k + N - 2) % N + 1
        
        ub[i, j, km2] +=  dzub[i, j, k]
        ub[i, j, km1] += -4dzub[i, j, k]
        ub[i, jm2, k] +=  dyub[i, j, k]
        ub[i, jm1, k] += -4dyub[i, j, k]
        ub[im2, j, k] +=  dxub[i, j, k]
        ub[im1, j, k] += -4dxub[i, j, k]
        ub[i, j, k]   +=  3 .* (
          dxub[i, j, k] + dyub[i, j, k] + dzub[i, j, k]
        )
      end
    end
  end
  yb = 0.0

  return ub, dxub, dyub, dzub

end # du3_mutation_adjoint
