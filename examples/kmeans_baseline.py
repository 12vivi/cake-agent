@cake.schedule()
def kmeans_assign(lm, X: LM.tma3d, C: LM.tma3d, CN: LM.gmem, OUT: LM.gmem):
    B, N, K, D = lm.dims("B", "N", "K", "D")
    BM, BN, STAGES, ACC = 128, 128, 2, 1
    TPB = lm.cdiv(N, BM)          # row tiles per batch
    nk = lm.cdiv(K, BN)           # centroid chunks
    lm.launch(grid=(TPB, B), warps=5)
    X_BYTES = BM * D * 2
    C_BYTES = BN * D * 2
    pool = lm.smem(X_BYTES + STAGES * C_BYTES)
    sx = pool.view(offset=0, shape=(BM, D), dtype=lm.bf16, swizzle="128B")
    sc = pool.view(offset=X_BYTES, shape=(BN, D), dtype=lm.bf16, stage=STAGES, swizzle="128B")
    acc = lm.tmem(cols=0, width=BN, shape=(BM, BN), dtype=lm.f32, stage=ACC)

    load = lm.role(warps=[0])
    comp = lm.role(warps=[1, 2, 3, 4])
    x_full = lm.barrier(prod=[load], cons=[comp])
    c_full = lm.barrier(count=STAGES, prod=[load], cons=[comp])
    c_free = lm.barrier(count=STAGES, prod=[comp], cons=[load], released=True)
    acc_full = lm.barrier(count=ACC, prod=[comp], cons=[comp])

    with load:
        b = lm.block_idx(1)
        m0 = lm.block_idx(0) * BM
        ti = 0
        sx.tma_load(X, coords=(b, m0, 0), barrier=x_full)
        for j in lm.range(0, nk):
            g = ti * nk + j
            lm.wait(c_free, stage=g % STAGES)
            sc.tma_load(C, coords=(b, j * BN, 0), stage=g % STAGES, barrier=c_full)

    with comp:
        best = lm.regs((BM,), lm.f32)
        arg = lm.regs((BM,), lm.i32)
        s = lm.regs((BM, 128), lm.f32)
        cn = lm.regs((128,), lm.f32)
        b = lm.block_idx(1)
        m0 = lm.block_idx(0) * BM
        ti = 0
        lm.fill(best, float('inf'))
        lm.fill(arg, 0)
        lm.wait(x_full)
        for j in lm.range(0, nk):
            g = ti * nk + j
            lm.wait(c_full, stage=g % STAGES)
            lm.mma(acc, sx, sc[g % STAGES], init=True)
            lm.commit(c_free, stage=g % STAGES)
            lm.commit(acc_full)
            lm.load(cn, CN, coords=(b, j * BN))
            lm.wait(acc_full)
            lm.tmem_load(s, acc)
            lm.ewise(s, "fma", s, -2.0, cn.row())
            lm.reduce(best, s, "argmin", idx=arg, index_offset=j * BN, valid=K - (j * BN))
        lm.store(OUT, coords=(b, m0), src=arg, valid=N - m0)
