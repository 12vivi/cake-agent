@cake.schedule()
def fmha_fwd(lm, Q: LM.tma3d, O: LM.tma2d):
    # Figure 3 of the paper (coordinates in numpy order): declare resources and roles;
    # the producer-consumer hand-off is an explicit barrier; addresses, phases and
    # warp identity are derived by lowering.
    pool = lm.smem(98304)
    smem_q = pool.view(offset=0, shape=(128, 128), dtype=lm.bf16, stage=3)
    tmem_acc = lm.tmem(cols=0, width=128, shape=(128, 128), dtype=lm.f32)
    load = lm.role(warps=[0])
    mma = lm.role(warps=[1])
    pipe = lm.pipeline(stages=3)
    q_full = lm.barrier(count=3, prod=[load], cons=[mma], init_count=1, pipeline=pipe)
    with load:
        for stage in lm.range(0, 3):
            smem_q.tma_load(Q, coords=(stage, 0, 0), stage=stage, barrier=q_full)
    with mma:
        for stage in lm.range(0, 3):
            lm.wait(q_full, stage=stage)
            lm.fence_proxy()
            lm.mma(tmem_acc, smem_q[stage], smem_q[stage], init=(stage == 0))
