if __name__ == "__main__":
    from .config import Configuration

    config = Configuration.parse_args()

    from .generator import DenseNetGenerator
    from .discriminator import Discriminator
