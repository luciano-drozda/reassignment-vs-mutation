function dU_ttgc_mutation(
  U         :: Matrix{Float64},
  Tetras    :: Matrix{Int64},
  Ve        :: Vector{Float64},
  Vj        :: Vector{Float64},
  Sk_X      :: Matrix{Float64},
  Sk_Y      :: Matrix{Float64},
  Sk_Z      :: Matrix{Float64},
  gamma_gas :: Float64,
  dt        :: Float64,
  beta      :: Float64,
  gamma     :: Float64
)

  ## 1st TTGC step

  # Initialize update to be computed
  # size (neq, nnode) -- (5, nnode)
  Up = zero(U)

  # Initialize (Mass-matrix * update) to be computed
  # size (neq, nnode) -- (5, nnode)
  MUp = zero(U)

  # Initialize nodal residuals to be computed
  # size (neq, nnode) -- (5, nnode)
  Rj = zero(U)

  # Initialize (partial) nodal residuals to be computed 
  # for 2nd step only
  # size (neq, nnode) -- (5, nnode)
  Rj2 = zero(U)

  # Loop over cells
  @inbounds for icell = 1:ncell

    # Initialize registers for cell-averaged values 
    # of primitive variables
    u = 0.0
    v = 0.0
    w = 0.0
    H = 0.0

    # Initialize register for pressure 
    # at nodes of the given cell
    # size (nvert,) -- (4,)
    p = zeros(4)

    # Initialize register for cell residual
    # size (neq,) -- (5,)
    VeRe = zeros(5)

    # Initialize register for cell flux jacobian 
    # (per direction)
    # size (neq,) -- (5,)
    AeReX = zeros(5)
    AeReY = zeros(5)
    AeReZ = zeros(5)

    # Initialize register for global indices 
    # of all nodes of the given cell
    inodes = zeros(Int, 4)

    # Loop over nodes of the given cell 
    # to compute `u`, `v`, `w` and `H`
    @inbounds for inode_local = 1:4

      # Get global node index
      inode = Tetras[inode_local, icell]
      
      # Store global node index in array
      inodes[inode_local] = inode

      # Get pressure at node
      p[inode_local] = (gamma_gas - 1) * (
        U[5,inode] - (
          U[2,inode]^2 + U[3,inode]^2 + U[4,inode]^2
        ) / 2U[1,inode]
      )

      # Add node contribution to cell-averaged values 
      # of primitive variables
      u += U[2, inode] / 4U[1, inode]
      v += U[3, inode] / 4U[1, inode]
      w += U[4, inode] / 4U[1, inode]
      H += (U[5, inode] + p[inode_local]) / 4U[1, inode]

    end # for inode_local

    # Loop again over nodes of the given cell 
    # to compute contributions to nodal residuals `Rj`
    @inbounds for inode_local = 1:4

      # Get global node index
      inode = inodes[inode_local]
      
      # Compute part of cell residual from current node
      # ieq == 1 (i.e., density)
      VeRe[1] = (-1/3) * (
        U[2, inode] * Sk_X[inode_local, icell] 
        + U[3, inode] * Sk_Y[inode_local, icell]
        + U[4, inode] * Sk_Z[inode_local, icell]
      )
      # ieq == 2 (i.e., x-momentum)
      VeRe[2] = (-1/3) * (
        (
          U[2, inode]^2 / U[1, inode] + p[inode_local]
        ) * Sk_X[inode_local, icell]
        + (
          U[2, inode] * U[3, inode] / U[1, inode]
        ) * Sk_Y[inode_local, icell]
        + (
          U[2, inode] * U[4, inode] / U[1, inode]
        ) * Sk_Z[inode_local, icell]
      )
      # ieq == 3 (i.e., y-momentum)
      VeRe[3] = (-1/3) * (
        (
          U[3, inode] * U[2, inode] / U[1, inode]
        ) * Sk_X[inode_local, icell]
        + (
          U[3, inode]^2 / U[1, inode] + p[inode_local]
        ) * Sk_Y[inode_local, icell]
        + (
          U[3, inode] * U[4, inode] / U[1, inode]
        ) * Sk_Z[inode_local, icell]
      )
      # ieq == 4 (i.e., z-momentum)
      VeRe[4] = (-1/3) * (
        (
          U[4, inode] * U[2, inode] / U[1, inode]
        ) * Sk_X[inode_local, icell]
        + (
          U[4, inode] * U[3, inode] / U[1, inode]
        ) * Sk_Y[inode_local, icell]
        + (
          U[4, inode]^2 / U[1, inode] + p[inode_local]
        ) * Sk_Z[inode_local, icell]
      )
      # ieq == 5 (i.e., energy)
      VeRe[5] = (-1/3) * (
        (U[2, inode] / U[1, inode]) * (
          U[5, inode] + p[inode_local]
        ) * Sk_X[inode_local, icell]
        + (U[3, inode] / U[1, inode]) * (
          U[5, inode] + p[inode_local]
        ) * Sk_Y[inode_local, icell]
        + (U[4, inode] / U[1, inode]) * (
          U[5, inode] + p[inode_local]
        ) * Sk_Z[inode_local, icell]
      )

      # Scatter the computed part of cell residual
      @inbounds for ieq = 1:5
        @inbounds for k = 1:4
          node = inodes[k]
          Rj[ieq, node] += (- dt/4) * (0.5 - gamma) * VeRe[ieq]
        end
      end

      # Build auxiliary differentials
      drho   = VeRe[1] / Ve[icell]
      drhou  = VeRe[2] / Ve[icell]
      drhov  = VeRe[3] / Ve[icell]
      drhow  = VeRe[4] / Ve[icell]
      dE     = VeRe[5] / Ve[icell]
      rhodu  = drhou - u * drho
      rhodv  = drhov - v * drho
      rhodw  = drhow - w * drho
      drhouu = u * drhou + u * rhodu
      drhovv = v * drhov + v * rhodv
      drhoww = w * drhow + w * rhodw
      drhouv = u * drhov + v * rhodu
      drhouw = u * drhow + w * rhodu
      drhovw = v * drhow + w * rhodv
      dP   = (gamma_gas - 1) * (
        dE - u * drhou - v * drhov - w * drhow + (
          u^2 + v^2 + w^2
        ) * drho / 2
      )
      drhoH  = dE + dP
      drhoHu = H * rhodu + u * drhoH
      drhoHv = H * rhodv + v * drhoH
      drhoHw = H * rhodw + w * drhoH

      # Compute part of cell flux jacobian from current node
      # ieq == 1 (i.e., density)
      AeReX[1] = drhou
      AeReY[1] = drhov
      AeReZ[1] = drhow
      
      # ieq == 2 (i.e., x-momentum)
      AeReX[2] = drhouu + dP
      AeReY[2] = drhouv
      AeReZ[2] = drhouw
      
      # ieq == 3 (i.e., y-momentum)
      AeReX[3] = drhouv
      AeReY[3] = drhovv + dP
      AeReZ[3] = drhovw
      
      # ieq == 4 (i.e., z-momentum)
      AeReX[4] = drhouw
      AeReY[4] = drhovw
      AeReZ[4] = drhoww + dP

      # ieq == 5 (i.e., energy)
      AeReX[5] = drhoHu
      AeReY[5] = drhoHv
      AeReZ[5] = drhoHw

      # Scatter the computed part of cell flux jacobian
      @inbounds for ieq = 1:5
        @inbounds for k = 1:4
          node = inodes[k]
          AeReSk = (
            AeReX[ieq] * Sk_X[k, icell] 
            + AeReY[ieq] * Sk_Y[k, icell]
            + AeReZ[ieq] * Sk_Z[k, icell]
          )
          Rj[ieq, node]  += (dt^2 / 3) * (beta * AeReSk)
          Rj2[ieq, node] += (dt^2 / 3) * (gamma * AeReSk)
        end
      end
      
    end # for inode_local

  end # for icell

  # Compute update via 2 iterations of the Jacobi algorithm
  # size (neq, nnode) -- (5, nnode)
  # First, evaluate initial guess `Up := Rj / Vj`
  # Loop over nodes and equations
  @inbounds for inode = 1:nnode
    @inbounds for ieq = 1:5
      Up[ieq, inode] = Rj[ieq, inode] / Vj[inode]
    end
  end
  # Invert mass matrix in Jacobi iterations
  # Loop over iterations
  @inbounds for _ = 1:2

    # Set (Mass-matrix * Up) to zero
    # MUp .= 0.
    @inbounds for inode = 1:nnode
      @inbounds for ieq = 1:5
        MUp[ieq, inode] = 0.
      end 
    end

    # Loop over cells and equations
    @inbounds for icell = 1:ncell

      # Set auxiliary constant
      c1 = Ve[icell] / 20

      @inbounds for ieq = 1:5
        
        # Get global nodes indices
        inode1 = Tetras[1, icell]
        inode2 = Tetras[2, icell]
        inode3 = Tetras[3, icell]
        inode4 = Tetras[4, icell]

        # Compute auxiliary variable
        summ = (
          Up[ieq, inode1]
          + Up[ieq, inode2]
          + Up[ieq, inode3]
          + Up[ieq, inode4]
        ) * c1
        
        # Add contribution
        MUp[ieq, inode1] += summ + Up[ieq, inode1] * c1
        MUp[ieq, inode2] += summ + Up[ieq, inode2] * c1
        MUp[ieq, inode3] += summ + Up[ieq, inode3] * c1
        MUp[ieq, inode4] += summ + Up[ieq, inode4] * c1

      end # for ieq
    end # for icell

    # Loop over nodes and equations
    @inbounds for inode = 1:nnode
      @inbounds for ieq = 1:5
        Up[ieq, inode] += (
          Rj[ieq, inode] - MUp[ieq, inode]
        ) / Vj[inode]
      end 
    end

  end # for _

  ## 2nd TTGC step

  # Update U
  # U .+= Up
  @inbounds for inode = 1:nnode
    @inbounds for ieq = 1:5
      U[ieq, inode] += Up[ieq, inode]
    end 
  end

  # Set `Rj` to zero
  # Rj .= 0.
  @inbounds for inode = 1:nnode
    @inbounds for ieq = 1:5
      Rj[ieq, inode] = 0.
    end 
  end

  # Loop over cells
  @inbounds for icell = 1:ncell

    # Initialize registers for cell-averaged values 
    # of primitive variables
    u = 0.0
    v = 0.0
    w = 0.0
    H = 0.0

    # Initialize register for pressure 
    # at nodes of the given cell
    # size (nvert,) -- (4,)
    p = zeros(4)

    # Initialize register for cell residual
    # size (neq,) -- (5,)
    VeRe = zeros(5)

    # Initialize register for global indices 
    # of all nodes of the given cell
    inodes = zeros(Int, 4)

    # Loop over nodes of the given cell 
    # to compute `u`, `v`, `w` and `H`
    @inbounds for inode_local = 1:4

      # Get global node index
      inode = Tetras[inode_local, icell]
      
      # Store global node index in array
      inodes[inode_local] = inode

      # Get pressure at node
      p[inode_local] = (gamma_gas - 1) * (
        U[5,inode] - (
          U[2,inode]^2 + U[3,inode]^2 + U[4,inode]^2
        ) / 2U[1,inode]
      )

      # Add node contribution to cell-averaged values 
      # of primitive variables
      u += U[2, inode] / 4U[1, inode]
      v += U[3, inode] / 4U[1, inode]
      w += U[4, inode] / 4U[1, inode]
      H += (U[5, inode] + p[inode_local]) / 4U[1, inode]

    end # for inode_local

    # Loop again over nodes of the given cell 
    # to compute contributions to nodal residuals `Rj`
    @inbounds for inode_local = 1:4

      # Get global node index
      inode = inodes[inode_local]
      
      # Compute part of cell residual from current node
      # ieq == 1 (i.e., density)
      VeRe[1] = (-1/3) * (
        U[2, inode] * Sk_X[inode_local, icell] 
        + U[3, inode] * Sk_Y[inode_local, icell]
        + U[4, inode] * Sk_Z[inode_local, icell]
      )
      # ieq == 2 (i.e., x-momentum)
      VeRe[2] = (-1/3) * (
        (
          U[2, inode]^2 / U[1, inode] + p[inode_local]
        ) * Sk_X[inode_local, icell]
        + (
          U[2, inode] * U[3, inode] / U[1, inode]
        ) * Sk_Y[inode_local, icell]
        + (
          U[2, inode] * U[4, inode] / U[1, inode]
        ) * Sk_Z[inode_local, icell]
      )
      # ieq == 3 (i.e., y-momentum)
      VeRe[3] = (-1/3) * (
        (
          U[3, inode] * U[2, inode] / U[1, inode]
        ) * Sk_X[inode_local, icell]
        + (
          U[3, inode]^2 / U[1, inode] + p[inode_local]
        ) * Sk_Y[inode_local, icell]
        + (
          U[3, inode] * U[4, inode] / U[1, inode]
        ) * Sk_Z[inode_local, icell]
      )
      # ieq == 4 (i.e., z-momentum)
      VeRe[4] = (-1/3) * (
        (
          U[4, inode] * U[2, inode] / U[1, inode]
        ) * Sk_X[inode_local, icell]
        + (
          U[4, inode] * U[3, inode] / U[1, inode]
        ) * Sk_Y[inode_local, icell]
        + (
          U[4, inode]^2 / U[1, inode] + p[inode_local]
        ) * Sk_Z[inode_local, icell]
      )
      # ieq == 5 (i.e., energy)
      VeRe[5] = (-1/3) * (
        (U[2, inode] / U[1, inode]) * (
          U[5, inode] + p[inode_local]
        ) * Sk_X[inode_local, icell]
        + (U[3, inode] / U[1, inode]) * (
          U[5, inode] + p[inode_local]
        ) * Sk_Y[inode_local, icell]
        + (U[4, inode] / U[1, inode]) * (
          U[5, inode] + p[inode_local]
        ) * Sk_Z[inode_local, icell]
      )

      # Scatter the computed part of cell residual
      @inbounds for ieq = 1:5
        @inbounds for k = 1:4
          node = inodes[k]
          Rj[ieq, node] += (- dt/4) * VeRe[ieq]
        end
      end
      
    end # for inode_local

  end # for icell

  # Accumulate flux cell jacobian related residual `Rj2` into `Rj`
  # Rj .+= Rj2
  @inbounds for inode = 1:nnode
    @inbounds for ieq = 1:5
      Rj[ieq, inode] += Rj2[ieq, inode]
    end 
  end

  # Compute update via 2 iterations of the Jacobi algorithm
  # size (neq, nnode) -- (5, nnode)
  # First, evaluate initial guess `Up := Rj / Vj`
  # Loop over nodes and equations
  @inbounds for inode = 1:nnode
    @inbounds for ieq = 1:5
      Up[ieq, inode] = (Rj[ieq, inode]) / Vj[inode]
    end
  end
  # Invert mass matrix in Jacobi iterations
  # Loop over iterations
  @inbounds for _ = 1:2

    # Set (Mass-matrix * Up) to zero
    # MUp .= 0.
    @inbounds for inode = 1:nnode
      @inbounds for ieq = 1:5
        MUp[ieq, inode] = 0.
      end 
    end

    # Loop over cells and equations
    @inbounds for icell = 1:ncell

      # Set auxiliary constant
      c1 = Ve[icell] / 20

      @inbounds for ieq = 1:5
        
        # Get global nodes indices
        inode1 = Tetras[1, icell]
        inode2 = Tetras[2, icell]
        inode3 = Tetras[3, icell]
        inode4 = Tetras[4, icell]

        # Compute auxiliary variable
        summ = (
          Up[ieq, inode1]
          + Up[ieq, inode2]
          + Up[ieq, inode3]
          + Up[ieq, inode4]
        ) * c1
        
        # Add contribution
        MUp[ieq, inode1] += summ + Up[ieq, inode1] * c1
        MUp[ieq, inode2] += summ + Up[ieq, inode2] * c1
        MUp[ieq, inode3] += summ + Up[ieq, inode3] * c1
        MUp[ieq, inode4] += summ + Up[ieq, inode4] * c1
  
      end # for ieq
    end # for icell

    # Loop over nodes and equations
    @inbounds for inode = 1:nnode
      @inbounds for ieq = 1:5
        Up[ieq, inode] += (
          Rj[ieq, inode] - MUp[ieq, inode]
        ) / Vj[inode]
      end 
    end

  end # for _

  # Return squared l2-norm of update to solution state `Up`
  return sum(abs2, Up)
  
end # dU_ttgc_mutation
