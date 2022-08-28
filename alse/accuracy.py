import torch, random
'''
Returns a tensor of N test points

            -pos_overlap should be a boolean tensor
             (Region of satisfaction)
            -candidates are all corresponding coordinates
'''
def gen_test_point_for_accuracy(pos_overlap, candidates, num_points, N, threshold=0.1):
    assert pos_overlap.dim() == candidates.dim()
    pos_overlap = pos_overlap.reshape(N,N,N)
    pos_mask = _get_boundary(pos_overlap).flatten().unsqueeze(-1)
    pos_mask = _clear_side(pos_mask.reshape(N,N,N)).flatten().unsqueeze(-1)
    pos_surface = _boundary_surface(candidates, pos_mask)
    # Must use double for _get_dist because of matrix multiplication, otherwise loss of precision
    dist = _get_dist(candidates.double(), pos_surface.double())
    dist_mask = _dist_mask(dist, threshold).unsqueeze(-1)
    # Convert to 1-D and unsqueeze
    pos_overlap = pos_overlap.flatten().unsqueeze(-1)  
                                                # Combined mask: in bound and in threshold
    all_in_bound_points = torch.masked_select(candidates, pos_overlap & dist_mask) # Returns a 1D tensor
    all_out_of_bound_points = torch.masked_select(candidates, ~pos_overlap & dist_mask)
    # Reshape to correct shape
    in_bound_cand = all_in_bound_points.reshape((pos_overlap & dist_mask).sum(), candidates.shape[1])
    out_of_bound_cand = all_out_of_bound_points.reshape((~pos_overlap & dist_mask).sum(), candidates.shape[1])
    # Select from available candidates
    in_bound_tpoints = in_bound_cand[torch.randint(in_bound_cand.shape[0], (num_points//2,))]
                                                                                # In case odd number points requested
    out_of_bound_tpoints = out_of_bound_cand[torch.randint(out_of_bound_cand.shape[0], (num_points//2 + num_points%2,))]
    return in_bound_tpoints, out_of_bound_tpoints, pos_surface

'''
Returns a boolean tensor representing the boundary of the posterior overlap
Input is the posterior overlap boolean tensor with dim N^d
'''
def _get_boundary(mask):
                        # Padd the right edge to prevent rollover to the left side
    padded = torch.nn.functional.pad(mask, (0,1),mode='constant')
                        # Shift the mask by 1 grid/pixel
    shifted = torch.roll(padded, 1)
            # XOR to find edge       # Exclude the padding
    return torch.logical_xor(mask, shifted[...,:-1])


def _boundary_surface(candidates, mask):
    return torch.masked_select(candidates, mask).reshape(mask.sum(), candidates.shape[1])

'''
Returns a tensor of the (minimum) distances to the boundary surface
Warning: Must use torch.double because of matrix multiplication
         otherwise catastrophic loss of precision
'''
def _get_dist(points, boundary):
    assert points.dtype == boundary.dtype == torch.double
    return torch.cdist(points, boundary).min(1).values 

'''
Returns a boolean tensor of region within threshold distance from the boundary surface
'''
def _dist_mask(distance, threshold):
    return distance < threshold

'''
Set all elements on all sides of a 3D tensor to 0
This will prevent false-positive
'''
def _clear_side(a):
    a[0] = a[-1] = a[...,0] = a[...,-1] = a[:,0] = a[:,-1] = 0
    return a