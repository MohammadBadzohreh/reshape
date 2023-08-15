import torch


def get_indices_list(map_type):
    if map_type == 1:
        return [[torch.arange(0, 25), torch.arange(25, 50)],
                [torch.arange(0, 2), torch.arange(2, 4)],
                [torch.arange(0, 2), torch.arange(2, 4)]]
    else:
        return [[torch.arange(0, 50, 2), torch.arange(1, 50, 2)],
                [torch.arange(0, 4, 2), torch.arange(1, 4, 2)],
                [torch.arange(0, 4, 2), torch.arange(1, 4, 2)]]


def external_reshape(x, map_type, device):
    # indices_list = get_indices_list(map_type)
    print("herer")
    # indices_list = [[torch.arange(0, 1), torch.arange(1, 2)],
    #                 [torch.arange(0, 1), torch.arange(1, 2)],
    #                 [torch.arange(0, 1), torch.arange(1, 2)]]

    # def tensor_reshape(x, indices):
    #     chunk = torch.index_select(
    #         torch.index_select(
    #             torch.index_select(x, 1, indices[0].to(device)),
    #             2, indices[1].to(device)),
    #         3, indices[2].to(device))
    #     return chunk

    # for i in range(2):
    #     for j in range(2):
    #         for k in range(2):
    #             chunk = tensor_reshape(
    #                 x, [indices_list[0][i], indices_list[1][j], indices_list[2][k]])

    #             if k == 0:
    #                 memory = chunk
    #             if k == 1 and j == 0:
    #                 memory2 = torch.stack((memory, chunk), dim=1).to(device)
    #             if k == 1 and j == 1 and i == 0:
    #                 memory3 = torch.stack((memory2, torch.stack(
    #                     (memory, chunk), dim=1)), dim=1).to(device)
    #             if k == 1 and j == 1 and i == 1:
    #                 x = torch.stack((memory3, torch.stack(
    #                     (memory2, torch.stack((memory, chunk), dim=1)), dim=1)), dim=1).to(device)
    # return x


def tensor_reshape2(x, indices, device):
    chunk = torch.index_select(
        torch.index_select(
            torch.index_select(x, 1, indices[0].to(device)),
            2, indices[1].to(device)),
        3, indices[2].to(device))
    return chunk


def external_reshape2(x, map_type, device):
    indices_list = get_indices_list(map_type)

    indices_combinations = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1)
    ]

    chunks = []
    for comb in indices_combinations:
        chunk = tensor_reshape2(
            x, [indices_list[0][comb[0]], indices_list[1][comb[1]], indices_list[2][comb[2]]], device)
        chunks.append(chunk)

    x = torch.stack((
        torch.stack((chunks[0], chunks[1]), dim=1),
        torch.stack((chunks[2], chunks[3]), dim=1)
    ), dim=1)

    x = torch.stack((x, torch.stack((
        torch.stack((chunks[4], chunks[5]), dim=1),
        torch.stack((chunks[6], chunks[7]), dim=1)
    ), dim=1)), dim=1)

    return x.to(device)


def tensor_reshape3(x, indices, device):
    # Use a loop to simplify repeated index_select
    for dim, index in enumerate(indices):
        x = torch.index_select(x, dim+1, index.to(device))
    return x


def external_reshape3(x, map_type, device):
    indices_list = get_indices_list(map_type)

    # Generate chunks using list comprehension for conciseness
    chunks = [tensor_reshape3(x, [indices_list[0][i], indices_list[1][j], indices_list[2][k]], device)
              for i in range(2) for j in range(2) for k in range(2)]

    # Reshape chunks systematically without hardcoding
    reshaped_chunks = []
    for i in range(0, len(chunks), 2):
        reshaped_chunks.append(torch.stack((chunks[i], chunks[i+1]), dim=1))

    result = torch.stack(reshaped_chunks, dim=1)
    return result.to(device)
