import os


LOGGER = None


class CsvLogger:
    def __init__(
        self,
        experiment: str,
        logs_folder: str = "logs",
    ):
        self.fname = os.path.join(
            logs_folder, os.path.split(experiment)[-1].replace(".ckpt", ".csv")
        )

    def initialize(self):
        with open(self.fname, "w") as f:
            f.write(self.header())
            f.write("\n")

    def header(self):
        return ";".join(
            [
                "profile",
                "QP",
                "ALF",
                "SAO",
                "DB",
                "intra",
                "ref_psnr",
                "psnr",
                "ref_crosslid",
                "crosslid",
            ]
        )

    def entries(self, data):
        entries = []
        for i in range(len(data["test_metadata"])):
            entries.append(
                ";".join(
                    [
                        "RA" if data["test_metadata"][i][0] == 1 else "AI",
                        data["test_metadata"][i][1] * 64,
                        data["test_metadata"][i][2],
                        data["test_metadata"][i][3],
                        data["test_metadata"][i][4],
                        data["test_metadata"][i][5],
                        data["test_ref_psnr"][i],
                        data["test_psnr"][i],
                        data["test_ref_crosslid"][i],
                        data["test_crosslid"][i],
                    ],
                )
            )
        return entries

    def log(self, data):
        with open(self.fname, "a") as f:
            f.writelines(self.entries(data))


def init(experiment: str, logs_folder: str = "logs"):
    global LOGGER
    LOGGER = CsvLogger(experiment, logs_folder)
    LOGGER.initialize()


def log(data):
    assert LOGGER is not None
    LOGGER.log(data)
