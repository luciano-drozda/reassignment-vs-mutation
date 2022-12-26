function dU_ttgc_reassignment(
  U         :: Matrix{Float64},
  Tetras    :: Matrix{Int64},
  Ve        :: Vector{Float64},
  Vj        :: Vector{Float64},
  Sk_X      :: Matrix{Float64},
  Sk_Y      :: Matrix{Float64},
  Sk_Z      :: Matrix{Float64},
  Sje_X     :: SparseMatrixCSC{Float64, Int64},
  Sje_Y     :: SparseMatrixCSC{Float64, Int64},
  Sje_Z     :: SparseMatrixCSC{Float64, Int64},
  S         :: SparseMatrixCSC{Float64, Int64},
  M         :: SparseMatrixCSC{Float64, Int64},
  gamma_gas :: Float64,
  dt        :: Float64,
  beta      :: Float64,
  gamma     :: Float64
)

  ## 1st TTGC step
    
  # Build auxiliary array `cell_U` 
  # w/ per-cell values of `U`
  # size (neq, nvert, ncell) -- (5, 4, ncell)
  cell_U = U[:, Tetras]

  # Build auxiliary array `cell_P` 
  # w/ per-cell values of pressure `P`
  # size (nvert, ncell) -- (4, ncell)
  cell_P = @. (gamma_gas - 1) * (
    cell_U[5, :, :] - (
      cell_U[2, :, :]^2 + cell_U[3, :, :]^2 + cell_U[4, :, :]^2
    ) / 2cell_U[1, :, :]
  )

  # Compute per-cell values of primitive variables
  # size (nvert, ncell) -- (4, ncell)
  cell_u = @. cell_U[2, :, :] / cell_U[1, :, :]
  cell_v = @. cell_U[3, :, :] / cell_U[1, :, :]
  cell_w = @. cell_U[4, :, :] / cell_U[1, :, :]
  cell_H = @. (cell_U[5, :, :] + cell_P) / cell_U[1, :, :]

  # Compute nodal fluxes `Fk_X/Y/Z`
  # size (neq, nvert, ncell) -- (5, 4, ncell)
  Fk_X = vcat(
    reshape(cell_U[2, :, :], 1, 4, :),
    reshape(cell_U[1, :, :] .* cell_u.^2 .+ cell_P, 1, 4, :),
    reshape(cell_U[1, :, :] .* cell_u .* cell_v, 1, 4, :),
    reshape(cell_U[1, :, :] .* cell_u .* cell_w, 1, 4, :),
    reshape(cell_U[1, :, :] .* cell_u .* cell_H, 1, 4, :),
  )
  Fk_Y = vcat(
    reshape(cell_U[3, :, :], 1, 4, :),
    reshape(cell_U[1, :, :] .* cell_v .* cell_u, 1, 4, :),
    reshape(cell_U[1, :, :] .* cell_v.^2 .+ cell_P, 1, 4, :),
    reshape(cell_U[1, :, :] .* cell_v .* cell_w, 1, 4, :),
    reshape(cell_U[1, :, :] .* cell_v .* cell_H, 1, 4, :),
  )
  Fk_Z = vcat(
    reshape(cell_U[4, :, :], 1, 4, :),
    reshape(cell_U[1, :, :] .* cell_w .* cell_u, 1, 4, :),
    reshape(cell_U[1, :, :] .* cell_w .* cell_v, 1, 4, :),
    reshape(cell_U[1, :, :] .* cell_w.^2 .+ cell_P, 1, 4, :),
    reshape(cell_U[1, :, :] .* cell_w .* cell_H, 1, 4, :),
  )

  # Compute cells residuals 
  # `VeRe := (-1/3) * sum_{k in Ke}( 
  #  (Fk_X * Sk_X) + (Fk_Y * Sk_Y) + (Fk_Z * Sk_Z) 
  # )`
  # size (neq, 1, ncell) -- (5, 1, ncell)
  VeRe = (-1/3) .* sum(
    Fk_X .* reshape(Sk_X, 1, 4, :)
    + Fk_Y .* reshape(Sk_Y, 1, 4, :) 
    + Fk_Z .* reshape(Sk_Z, 1, 4, :); dims=2
  )

  # Compute cell-averaged primitive variables
  # size (1, ncell)
  u = sum(cell_u, dims=1) ./ 4
  v = sum(cell_v, dims=1) ./ 4
  w = sum(cell_w, dims=1) ./ 4
  H = sum(cell_H, dims=1) ./ 4

  # Build auxiliary differentials
  # size (1, ncell)
  drho   = VeRe[1, :, :] ./ reshape(Ve, 1, :)  # `- drho / dt`
  drhou  = VeRe[2, :, :] ./ reshape(Ve, 1, :)
  drhov  = VeRe[3, :, :] ./ reshape(Ve, 1, :)
  drhow  = VeRe[4, :, :] ./ reshape(Ve, 1, :)
  dE   = VeRe[5, :, :] ./ reshape(Ve, 1, :)
  rhodu  = @. drhou - u * drho
  rhodv  = @. drhov - v * drho
  rhodw  = @. drhow - w * drho
  drhouu = @. u * drhou + u * rhodu
  drhovv = @. v * drhov + v * rhodv
  drhoww = @. w * drhow + w * rhodw
  drhouv = @. u * drhov + v * rhodu
  drhouw = @. u * drhow + w * rhodu
  drhovw = @. v * drhow + w * rhodv
  dP   = @. (gamma_gas - 1) * (
    dE - u * drhou - v * drhov - w * drhow + (
      u^2 + v^2 + w^2
    ) * drho / 2
  )
  drhoH  = @. dE + dP
  drhoHu = @. H * rhodu + u * drhoH
  drhoHv = @. H * rhodv + v * drhoH
  drhoHw = @. H * rhodw + w * drhoH

  # Compute cell flux jacobians `AeRe_X/Y/Z`
  # size (neq, ncell) -- (5, ncell)
  AeRe_X = vcat(
    drhou,
    drhouu + dP,
    drhouv,
    drhouw,
    drhoHu
  )
  AeRe_Y = vcat(
    drhov,
    drhouv,
    drhovv + dP,
    drhovw,
    drhoHv
  )
  AeRe_Z = vcat(
    drhow,
    drhouw,
    drhovw,
    drhoww + dP,
    drhoHw
  )

  # Compute 1st part of nodal residuals, namely, `Lj`
  # size (neq, nnode)
  VeRe = dropdims(VeRe; dims=2) # (5, ncell)
  Lj   = (S * VeRe')' ./ 4

  # Compute 2nd part of nodal residuals, namely, `LLj`
  # size (neq, nnode)
  LLj = (
    Sje_X * AeRe_X' .+ Sje_Y * AeRe_Y' .+ Sje_Z * AeRe_Z'
  )' ./ 3

  # Compute nodal residuals `Rj`
  # size (neq, nnode) -- (5, nnode)
  Rj = -dt .* (0.5 - gamma) .* Lj + beta .* dt^2 .* LLj

  # Compute update via 2 iterations of the Jacobi algorithm
  # size (neq, nnode) -- (5, nnode)
  Up = Rj ./ Vj'
  @inbounds for _ = 1:2
    Up += (Rj' - M * Up')' ./ Vj'
  end

  ## 2nd TTGC step

  # Build auxiliary array `cell_Ut` 
  # w/ per-cell values of `Ut := U + Up`
  # size (neq, nvert, ncell) -- (5, 4, ncell)
  Ut      = U + Up
  cell_Ut = Ut[:, Tetras]

  # Build auxiliary array `cell_Pt` 
  # w/ per-cell values of pressure `P`
  # size (nvert, ncell) -- (4, ncell)
  cell_Pt = @. (gamma_gas - 1) * (
    cell_Ut[5, :, :] - (
      cell_Ut[2, :, :]^2 + cell_Ut[3, :, :]^2 + cell_Ut[4, :, :]^2
    ) / 2cell_Ut[1, :, :]
  )

  # Compute per-cell values of primitive variables
  # size (nvert, ncell) -- (4, ncell)
  cell_ut = @. cell_Ut[2, :, :] / cell_Ut[1, :, :]
  cell_vt = @. cell_Ut[3, :, :] / cell_Ut[1, :, :]
  cell_wt = @. cell_Ut[4, :, :] / cell_Ut[1, :, :]
  cell_Ht = @. (cell_Ut[5, :, :] + cell_Pt) / cell_Ut[1, :, :]

  # Compute nodal fluxes `Fk_X/Y/Z`
  # size (neq, nvert, ncell) -- (5, 4, ncell)
  Fk_X = vcat(
    reshape(cell_Ut[2, :, :], 1, 4, :),
    reshape(cell_Ut[1, :, :] .* cell_ut.^2 .+ cell_Pt, 1, 4, :),
    reshape(cell_Ut[1, :, :] .* cell_ut .* cell_vt, 1, 4, :),
    reshape(cell_Ut[1, :, :] .* cell_ut .* cell_wt, 1, 4, :),
    reshape(cell_Ut[1, :, :] .* cell_ut .* cell_Ht, 1, 4, :),
  )
  Fk_Y = vcat(
    reshape(cell_Ut[3, :, :], 1, 4, :),
    reshape(cell_Ut[1, :, :] .* cell_vt .* cell_ut, 1, 4, :),
    reshape(cell_Ut[1, :, :] .* cell_vt.^2 .+ cell_Pt, 1, 4, :),
    reshape(cell_Ut[1, :, :] .* cell_vt .* cell_wt, 1, 4, :),
    reshape(cell_Ut[1, :, :] .* cell_vt .* cell_Ht, 1, 4, :),
  )
  Fk_Z = vcat(
    reshape(cell_Ut[4, :, :], 1, 4, :),
    reshape(cell_Ut[1, :, :] .* cell_wt .* cell_ut, 1, 4, :),
    reshape(cell_Ut[1, :, :] .* cell_wt .* cell_vt, 1, 4, :),
    reshape(cell_Ut[1, :, :] .* cell_wt.^2 .+ cell_Pt, 1, 4, :),
    reshape(cell_Ut[1, :, :] .* cell_wt .* cell_Ht, 1, 4, :),
  )

  # Compute cells residuals 
  # `VeRe := (-1/3) * sum_{k in Ke}( 
  #  (Fk_X * Sk_X) + (Fk_Y * Sk_Y) + (Fk_Z * Sk_Z) 
  # )`
  # size (neq, 1, ncell) -- (5, 1, ncell)
  VeRe = (-1/3) .* sum(
    Fk_X .* reshape(Sk_X, 1, 4, :)
    + Fk_Y .* reshape(Sk_Y, 1, 4, :) 
    + Fk_Z .* reshape(Sk_Z, 1, 4, :); dims=2
  )

  # Compute 1st part of nodal residuals, namely, `Lj`
  # size (neq, nnode)
  VeRe = dropdims(VeRe; dims=2) # (5, ncell)
  Lj   = (S * VeRe')' ./ 4

  # Compute nodal residuals `Rj`
  # size (neq, nnode) -- (5, nnode)
  Rj = -dt .* Lj + gamma .* dt^2 .* LLj

  # Compute update via 2 iterations of the Jacobi algorithm
  # size (neq, nnode) -- (5, nnode)
  Up = Rj ./ Vj'
  @inbounds for _ = 1:2
    Up += (Rj' - M * Up')' ./ Vj'
  end

  # Return squared l2-norm of update to solution state `Up`
  return sum(abs2, Up)

end # dU_ttgc_reassignment
