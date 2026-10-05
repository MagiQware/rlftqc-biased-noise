import numpy as np

from dataclasses import dataclass
from typing import NamedTuple


class Neighbors(NamedTuple):
    north: int | None
    east: int | None
    south: int | None
    west: int | None


@dataclass(frozen=True)
class Cell:
    # cell type is either 'bulk' or 'boundary' enum
    cell_type: str 
    measure_qubit: int
    data_qubits: Neighbors

    # Mask order: center, north, east, south, west
    action_mask: tuple[bool, bool, bool, bool, bool]


def get_mapping():
    rows = 5
    cols = 5
    corners = [(0, 0), (1, 0), (0, cols - 1),  (0, cols - 2), (rows - 1, 0), (rows - 1, 1), (rows - 1, cols - 1), (rows - 2, cols - 1)]
    mapping = []
    qubit_index = 0
    for i in range(rows):
        for j in range(cols):
            if (i, j) not in corners:
                mapping.append([i, j])
                qubit_index += 1

    return np.array(mapping)

def build_cells():
    rows = 5
    cols = 5
    mapping = get_mapping()
    print(mapping)
    index_grid = -np.ones((rows, cols), dtype=int)
    for idx, (i, j) in enumerate(mapping):
        index_grid[i, j] = idx
    print(f"index_grid:\n {index_grid}")

    full_cells = [3, 7, 9, 13]
    cells = []
    for cell_idx in full_cells:
        position = mapping[cell_idx]
        cell = Cell(
            cell_type='bulk',
            measure_qubit=cell_idx,
            data_qubits=Neighbors(
                north=index_grid[position[0] - 1, position[1]] if position[0] > 0 else None,
                east=index_grid[position[0], position[1] + 1] if position[1] < index_grid.shape[1] - 1 else None,
                south=index_grid[position[0] + 1, position[1]] if position[0] < index_grid.shape[0] - 1 else None,
                west=index_grid[position[0], position[1] - 1] if position[1] > 0 else None
            ),
            action_mask=(True, True, True, True, True)
        )
        cells.append(cell)

    boundary_cells = [0, 5, 16, 11]
    for cell_idx in boundary_cells:
        position = mapping[cell_idx]
        cell = Cell(
            cell_type='boundary',
            measure_qubit=cell_idx,
            data_qubits=Neighbors(
                north=index_grid[position[0] - 1, position[1]] if (position[0] > 0 and index_grid[position[0] - 1, position[1]] != -1) else None,
                east=index_grid[position[0], position[1] + 1] if (position[1] < index_grid.shape[1] - 1 and index_grid[position[0], position[1] + 1] != -1) else None,
                south=index_grid[position[0] + 1, position[1]] if (position[0] < index_grid.shape[0] - 1 and index_grid[position[0] + 1, position[1]] != -1) else None,
                west=index_grid[position[0], position[1] - 1] if (position[1] > 0 and index_grid[position[0], position[1] - 1] != -1) else None
            ),
            # action mask should set nones to false
            action_mask=(
                True,
                True if position[0] > 0 and index_grid[position[0] - 1, position[1]] != -1 else False,
                True if position[1] < index_grid.shape[1] - 1 and index_grid[position[0], position[1] + 1] != -1 else False,
                True if position[0] < index_grid.shape[0] - 1 and index_grid[position[0] + 1, position[1]] != -1 else False,
                True if position[1] > 0 and index_grid[position[0], position[1] - 1] != -1 else False
            )
        )
        cells.append(cell)

    return cells


def get_target():

    mapping = get_mapping()

    target_cells = build_cells()

    print(f"mapping: {mapping}")
    print(f"target_cells: {target_cells}")

    target = []

    n = 17
    logical_indices = [1, 4, 10]
    logical = 'I' * n
    paulis = ['X', 'Z']
    for i,qubit in enumerate(logical_indices):
        logical = logical[:qubit] + paulis[i % len(paulis)] + logical[qubit+1:]

    target.append(logical)
    print(f"n: {n}")
    measure_qubits = []
    for i, cell in enumerate(target_cells):
        measure_qubit = cell.measure_qubit
        data_qubits = cell.data_qubits
        action_mask = np.array(cell.action_mask)
        stab = 'I' * n
        for data_qubit, mask in zip([data_qubits.north, data_qubits.south], action_mask[[1,3]]):
            print(f"data_qubit: {data_qubit}, mask: {mask}")
            if mask:
                stab = stab[:data_qubit] + 'Z' + stab[data_qubit+1:]

        for data_qubit, mask in zip([data_qubits.east, data_qubits.west], action_mask[[2, 4]]):
            print(f"data_qubit: {data_qubit}, mask: {mask}")
            if mask:
                stab = stab[:data_qubit] + 'X' + stab[data_qubit+1:]

        # measure qubit must get back positive X by measuring stabilizer
        stab = stab[:measure_qubit] + 'X' + stab[measure_qubit+1:]
        
        target.append(stab)
        print(f"stab for cell {i}: {stab}")
        measure_qubits.append(measure_qubit)

    for measure_qubit in measure_qubits:
        print(f"measure_qubit: {measure_qubit}")
        stabilizer = 'I' * n
        stabilizer = stabilizer[:measure_qubit] + 'X' + stabilizer[measure_qubit+1:]
        target.append(stabilizer)


    for stab in target:
        print(f"stab: {stab}")

    return target