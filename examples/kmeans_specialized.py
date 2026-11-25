@cake.schedule()
def kmeans_assign(lm, X: LM.tma3d, C: LM.tma3d, CN: LM.gmem, OUT: LM.gmem):
    B, N, K, D = lm.dims("B", "N", "K", "D")
    BM, BN, STAGES, ACC = 128, 128, 2, 2
    TPB = lm.cdiv(N, BM)          # row tiles per batch
    nk = lm.cdiv(K, BN)           # centroid chunks
    lm.launch(grid=(TPB, B), warps=6)
    X_BYTES = BM * D * 2
    C_BYTES = BN * D * 2
    pool = lm.smem(X_BYTES + STAGES * C_BYTES)
    sx = pool.view(offset=0, shape=(BM, D), dtype=lm.bf16, swizzle="128B")
    sc = pool.view(offset=X_BYTES, shape=(BN, D), dtype=lm.bf16, stage=STAGES, swizzle="128B")
    acc = lm.tmem(cols=0, width=BN, shape=(BM, BN), dtype=lm.f32, stage=ACC)

    load = lm.role(warps=[0])
    mma = lm.role(warps=[1])
    epi = lm.role(warps=[2, 3, 4, 5])
    x_full = lm.barrier(prod=[load], cons=[mma])
    c_full = lm.barrier(count=STAGES, prod=[load], cons=[mma])
    c_free = lm.barrier(count=STAGES, prod=[mma], cons=[load], released=True)
    acc_full = lm.barrier(count=ACC, prod=[mma], cons=[epi])
    acc_free = lm.barrier(count=ACC, prod=[epi], cons=[mma], released=True)

    with load:
        b = lm.block_idx(1)
        m0 = lm.block_idx(0) * BM
        ti = 0
        sx.tma_load(X, coords=(b, m0, 0), barrier=x_full)
        for j in lm.range(0, nk):
            g = ti * nk + j
            lm.wait(c_free, stage=g % STAGES)
            sc.tma_load(C, coords=(b, j * BN, 0), stage=g % STAGES, barrier=c_full)

    with mma:
        b = lm.block_idx(1)
        m0 = lm.block_idx(0) * BM
        ti = 0
        lm.wait(x_full)
        for j in lm.range(0, nk):
            g = ti * nk + j
            lm.wait(c_full, stage=g % STAGES)
            lm.wait(acc_free, stage=g % ACC)
            lm.mma(acc[g % ACC], sx, sc[g % STAGES], init=True)
            lm.commit(c_free, stage=g % STAGES)
            lm.commit(acc_full, stage=g % ACC)

    with epi:
        best = lm.regs((BM,), lm.f32)
        arg = lm.regs((BM,), lm.i32)
        s = lm.regs((BM, 128), lm.f32)
        cn = lm.regs((128,), lm.f32)
        b = lm.block_idx(1)
        m0 = lm.block_idx(0) * BM
        ti = 0
        lm.fill(best, float('inf'))
        lm.fill(arg, 0)
        for j in lm.range(0, nk):
            g = ti * nk + j
            lm.load(cn, CN, coords=(b, j * BN))
            lm.wait(acc_full, stage=g % ACC)
            lm.tmem_load(s, acc[g % ACC])
            lm.arrive(acc_free, stage=g % ACC)
            lm.ewise(s, "fma", s, -2.0, cn.row())
            lm.reduce(best, s, "argmin", idx=arg, index_offset=j * BN, valid=K - (j * BN))
        lm.store(OUT, coords=(b, m0), src=arg, valid=N - m0)
