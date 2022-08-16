import torch, random
'''
            -pos_overlap should be a boolean tensor
             (Region of satisfaction for modelA & modelB & modelC ...)
            -candidates are all corresponding coordinates
'''
def gen_test_point_for_accuracy(pos_overlap, candidates, num_points):
    assert pos_overlap.dim() == candidates.dim()
    # Separate candidates
    masked_in = torch.masked_select(candidates, pos_overlap) # Returns a 1D tensor
    masked_out = torch.masked_select(candidates, ~pos_overlap)
    # Reshape to correct shape
    in_bound_cand = masked_in.reshape(pos_overlap.sum(), candidates.shape[1])
    out_of_bound_cand = masked_out.reshape((~pos_overlap).sum(), candidates.shape[1])
    # Select from available candidates
    in_bound_tpoints = in_bound_cand[torch.randint(in_bound_cand.shape[0], (num_points//2,))]
                                                                                # In case odd number points requested
    out_of_bound_tpoints = out_of_bound_cand[torch.randint(in_bound_cand.shape[0], (num_points//2 + num_points%2,))]
    return in_bound_tpoints, out_of_bound_tpoints


