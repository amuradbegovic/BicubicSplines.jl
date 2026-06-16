module BicubicSplines

using LinearAlgebra

export BicubicSpline

mutable struct BicubicSpline
    x::AbstractVector
    y::AbstractVector
    dx::AbstractVector
    dy::AbstractVector
    u::AbstractMatrix

    p::AbstractMatrix
    q::AbstractMatrix
    s::AbstractMatrix

    M::Int64
    N::Int64

    A::Matrix{Matrix}

    function BicubicSpline(x::AbstractVector, y::AbstractVector,
                           u::AbstractMatrix,
                           edge_p::AbstractMatrix=zeros(2,length(y)),
                           edge_q::AbstractMatrix=zeros(length(x),2),
                           edge_s::AbstractMatrix=zeros(2,2))
        if length(x) != size(u,1) || length(y) != size(u,2) ||
           size(edge_p,1) != 2 || size(edge_p,2) != length(y) ||
           size(edge_q,1) != length(x) || size(edge_q, 2) != 2 ||
           size(edge_s,1) != 2 || size(edge_s,2) != 2
            error("Bad dimensions")
        end

        M = length(x)
        N = length(y)

        p = zeros(M, N)
        p[1,:] = edge_p[1,:]
        p[M,:] = edge_p[2,:]

        q = zeros(M, N)
        q[:,1] = edge_q[:,1]
        q[:,N] = edge_q[:,2]

        s = zeros(M, N)
        s[1,1] = edge_s[1,1]
        s[1,N] = edge_s[1,2]
        s[M,1] = edge_s[2,1]
        s[M,N] = edge_s[2,2]

        dx = [x[i+1] - x[i] for i=1:M-1]
        dy = [y[j+1] - y[j] for j=1:N-1]

        rho = [2*(dx[i] + dx[i+1]) for i=1:M-2]
        sigma = [2*(dy[j] + dy[j+1]) for j=1:N-2]

        alpha = [3 * ((dx[i]/dx[i+1])*(u[i+2,j] - u[i+1,j]) + (dx[i+1]/dx[i])*(u[i+1,j] - u[i,j])) for i=1:M-2,j=1:N]
        for j in 1:N
            alpha[1,j] -= dx[2]*p[1,j]
            alpha[M-2,j] -= dx[M-2]*p[M,j]
        end

        beta = [3 * ((dx[i]/dx[i+1])*(q[i+2,j] - q[i+1,j]) + (dx[i+1]/dx[i])*(q[i+1,j] - q[i,j])) for i=1:M-2,j=1:N]
        beta[1,1] -= dx[2]*s[1,1]
        beta[1,N] -= dx[2]*s[1,N]
        beta[M-2,1] -= dx[M-2]*s[M,1]
        beta[M-2,N] -= dx[M-2]*s[M,N]

        gamma = [3 * ((dy[j]/dy[j+1])*(u[i,j+2] - u[i,j+1]) + (dy[j+1]/dy[j])*(u[i,j+1] - u[i,j])) for i=1:M, j=1:N-2]
        for i in 1:M
            gamma[i,1] -= dy[2]*q[i,1]
            gamma[i,N-2] -= dy[N-2]*q[i,N]
        end

        delta = [3 * ((dy[j]/dy[j+1])*(p[i,j+2] - p[i,j+1]) + (dy[j+1]/dy[j])*(p[i,j+1] - p[i,j])) for i=1:M, j=1:N-2]
        for i in 1:M
            delta[i,1] -= dy[2]*s[i,1]
            delta[i,N-2] -= dy[N-2]*s[i,N]
        end

        rho_lu = lu(Tridiagonal(dx[3:M-1], rho, dx[1:M-3]))

        p[2:M-1,:] = rho_lu \ alpha
        s[2:M-1,1] = rho_lu \ beta[:,1]
        s[2:M-1,N] = rho_lu \ beta[:,N]

        sigma_lu = lu(Tridiagonal(dy[3:N-1], sigma, dy[1:N-3]))

        q[:,2:N-1] = (sigma_lu' \ gamma')'
        s[:,2:N-1] = (sigma_lu' \ delta')'

        A = Matrix{Matrix}(undef, M-1,N-1)

        new(x, y, dx, dy, u, p, q, s, M, N, A)
    end
end

function locate(bs::BicubicSpline, x0, y0)
    i = searchsortedlast(bs.x, x0)
    j = searchsortedlast(bs.y, y0)

    i = clamp(i, 1, bs.M - 1)
    j = clamp(j, 1, bs.N - 1)

    return i, j
end

function (bs::BicubicSpline)(x0, y0)
    i, j = locate(bs, x0, y0)

    if !isassigned(bs.A, i, j)
        B(h) = [1 0 0 0;
                0 1 0 0;
                -3/h^2 -2/h 3/h^2 -1/h;
                2/h^3 1/h^2 -2/h^3 1/h^2]
        K(i,j) = [bs.u[i,j] bs.q[i,j] bs.u[i,j+1] bs.q[i,j+1];
                  bs.p[i,j] bs.s[i,j] bs.p[i,j+1] bs.s[i,j+1];
                  bs.u[i+1,j] bs.q[i+1,j] bs.u[i+1,j+1] bs.q[i+1,j+1];
                  bs.p[i+1,j] bs.s[i+1,j] bs.p[i+1,j+1] bs.s[i+1,j+1]]

        bs.A[i,j] = B(bs.dx[i])*K(i,j)*B(bs.dy[j])'
    end

    z = 0.0
    dx0 = x0 - bs.x[i]
    dy0 = y0 - bs.y[j]
    for m in 1:4
        for n in 1:4
            z += bs.A[i,j][m,n] * ((dx0)^(m-1)) * ((dy0)^(n-1))
        end
    end

    return z
end

end # module BicubicSplines
