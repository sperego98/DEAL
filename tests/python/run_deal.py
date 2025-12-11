from deal import DataConfig, DEALConfig, FlareConfig, DEAL

input_path = '../data/traj.xyz'

cutoff = 5
deal_thresholds = [0.05,0.1,0.15]

for deal_threshold in deal_thresholds:
    output_prefix = f'deal_{deal_threshold:.3f}'

    data_cfg = DataConfig(
        files=input_path,
    )

    deal_cfg = DEALConfig(
        threshold=deal_threshold,
        output_prefix=output_prefix,
        verbose='debug',
    )

    flare_cfg = FlareConfig(cutoff=cutoff)

    deal = DEAL(data_cfg, deal_cfg, flare_cfg)
    # uncomment to run
    deal.run()
