import logging
from collections import defaultdict

class NerData():
    def __init__(self, path):
        self.data = self.load(path)

    def load(self, path):
        data = []

        with open(path, "r", encoding="utf-8") as fp:
            buffer = defaultdict(list)
            is_num = False

            for line in fp.readlines():
                line = line.replace("\n", "")

                if not line:
                    continue

                if not is_num and line.startswith("## "):
                    is_num = True

                    if buffer:
                        data.append(buffer)
                        buffer = defaultdict(list)

                elif is_num and line.startswith("## "):
                    is_num = False
                    buffer["sentence"] = line.replace("## ", "")
                    buffer["chars"] = []

                else:
                    c_buffer = line.split("\t")

                    if c_buffer[0] == " ":
                        continue

                    buffer["chars"].append({"char": c_buffer[0], "label": c_buffer[1]})

            # TODO: Log 남기기
            # FIXME: Hello
            logging.info(f"Loaded NER dataset, Data Size : {len(data)}")

        return data