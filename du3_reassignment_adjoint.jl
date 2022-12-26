function du3_reassignment_adjoint(
  u   :: Array{Float64, 3}, 
  dxu :: Array{Float64, 3}, 
  dyu :: Array{Float64, 3}, 
  dzu :: Array{Float64, 3}
)

  u_im1 = circshift(u, (1, 0, 0))
  u_im2 = circshift(u, (2, 0, 0))
  dxu  += u_im2 - 4u_im1 + 3u

  u_jm1 = circshift(u, (0, 1, 0))
  u_jm2 = circshift(u, (0, 2, 0))
  dyu  += u_jm2 - 4u_jm1 + 3u

  u_km1 = circshift(u, (0, 0, 1))
  u_km2 = circshift(u, (0, 0, 2))
  dzu  += u_km2 - 4u_km1 + 3u

  yb   = 1.0
  ub   = zero(u)
  dxub = 2yb .* dxu
  dyub = 2yb .* dyu
  dzub = 2yb .* dzu
  ub += (
    circshift(dxub, (-2, 0, 0)) 
    + circshift(-4dxub, (-1, 0, 0)) 
    + 3dxub 
    + circshift(dyub, (0, -2, 0)) 
    + circshift(-4dyub, (0, -1, 0)) 
    + 3dyub 
    + circshift(dzub, (0, 0, -2)) 
    + circshift(-4dzub, (0, 0, -1)) 
    + 3dzub
  )
  yb = 0.0

  return ub, dxub, dyub, dzub

end # du3_reassignment_adjoint
