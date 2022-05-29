class cell:
    def __init__(self, num) -> None:
        self.aaa = num


a = [cell(1), cell(2), cell(3)]

print(min(a, key=lambda x: x.aaa).aaa)
