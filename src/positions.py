import numpy as np
from itertools import product
from abc import ABC, abstractmethod


class AbstractPositions(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def cast_to_tuple(self, x):
        if isinstance(x, (list, np.ndarray)):
            return tuple(x)
        elif isinstance(x, tuple):
            return x
        else:  # For scalars, wrap it in a tuple
            return (x,)


class Positions(AbstractPositions):
    def __init__(self, positions):
        self.positions = positions

    def __call__(self, *args):
        return self.positions

    def to_dict(self):
        return {"positions": self.positions}


class GridPositions(AbstractPositions):
    def __init__(self, domain):
        self.domain = self.cast_to_tuple(domain)

    def __call__(self, shape):
        shape = self.cast_to_tuple(shape)
        if len(shape) != len(self.domain):
            raise ValueError("Shape and size must have the same length")
        grids = []
        for s, n in zip(self.domain, shape):
            offset = (s / n) / 2  # Offset to center the grid
            grids.append(np.linspace(0 + offset, s + offset, n, endpoint=False))
        positions = list(product(*grids))
        return positions

    def to_dict(self):
        return {"domain": self.domain}


class RandomPositions(AbstractPositions):
    def __init__(self, domain):
        self.domain = self.cast_to_tuple(domain)

    def __call__(self, shape):
        shape = self.cast_to_tuple(shape)
        total_positions = np.prod(shape)
        positions = []
        for s in self.domain:
            positions.append(np.random.uniform(0, s, total_positions))
        positions = list(zip(*positions))
        return positions

    def to_dict(self):
        return {"domain": self.domain}


class ClusteredPositions(AbstractPositions):
    def __init__(self, center, radius):
        self.center = self.cast_to_tuple(center)
        self.radius = radius

    def __call__(self, shape):
        shape = self.cast_to_tuple(shape)
        total_positions = np.prod(shape)
        # Generate random angles uniformly between 0 and 2π
        theta = np.random.uniform(0, 2 * np.pi, total_positions)
        # Generate random radii with uniform distribution over disc area
        r = self.radius * np.sqrt(np.random.uniform(0, 1, total_positions))
        # Convert polar coordinates to Cartesian coordinates
        x = self.center[0] + r * np.cos(theta)
        y = self.center[1] + r * np.sin(theta)
        positions = list(zip(x, y))
        return positions

    def to_dict(self):
        return {"center": self.center, "radius": self.radius}
