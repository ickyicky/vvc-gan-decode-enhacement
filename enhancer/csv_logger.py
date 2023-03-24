import os


LOGGER = None


class CsvLogger:
    def __init__(
        self,
        experiment: str,
        logs_folder: str = "logs",
    ):
        self.logs_folder = logs_folder
        self.fname = os.path.join(
            logs_folder, os.path.split(experiment)[-1].replace(".ckpt", ".csv")
        )

    def initialize(self):
        if not os.path.exists(self.logs_folder):
            os.mkdir(self.logs_folder)

        with open(self.fname, "w") as f:
            f.write(self.header())

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
                "ref_ssim",
                "ssim",
                "ref_crosslid",
                "crosslid",
            ]
        )

    def entries(self, data):
        entries = []

        metadata = data["test_metadata"].cpu()
        ref_psnr = data["test_ref_psnr"]
        psnr = data["test_psnr"]
        ref_ssim = data["test_ref_ssim"]
        ssim = data["test_ssim"]
        ref_crosslid = data["test_ref_crosslid"]
        crosslid = data["test_crosslid"]

        for i in range(len(metadata)):
            mt = metadata[i].view(-1)
            entries.append(
                ";".join(
                    [
                        "RA" if mt[0] == 1 else "AI",
                        str(int(mt[1] * 64)),
                        str(int(mt[2])),
                        str(int(mt[3])),
                        str(int(mt[4])),
                        str(int(mt[5])),
                        str(float(ref_psnr[i].cpu())),
                        str(float(psnr[i].cpu())),
                        str(float(ref_ssim[i].cpu())),
                        str(float(ssim[i].cpu())),
                        str(float(ref_crosslid[i])),
                        str(float(crosslid[i])),
                    ],
                )
            )

        return entries

    def log(self, data):
        with open(self.fname, "a") as f:
            f.write("\n")
            f.write("\n".join(self.entries(data)))


def init(experiment: str, logs_folder: str = "logs"):
    global LOGGER
    LOGGER = CsvLogger(experiment, logs_folder)
    LOGGER.initialize()


def log(data):
    assert LOGGER is not None
    LOGGER.log(data)
