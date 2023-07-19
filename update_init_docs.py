"""Update the docstring in __init__.py with the README.md file."""


with open("README.md") as f:
    readme = list(f.readlines())

while not readme[0].startswith("[!"):
    readme.pop(0)
while readme[0].startswith("[!"):
    readme.pop(0)
while len(readme[0].strip()) == 0:
    readme.pop(0)

readme = [x.strip() + "\n" for x in readme]
license_index = readme.index("## License\n")
readme = readme[:license_index]

with open("proteinflow/__init__.py") as f:
    init = list(f.readlines())

while not init[0].startswith("__pdoc__"):
    init.pop(0)

init = ['"""\n'] + readme + ['"""\n'] + init
with open("proteinflow/__init__.py", "w") as f:
    f.writelines(init)
