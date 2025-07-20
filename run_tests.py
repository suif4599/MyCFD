import tests

for test in dir(tests):
    if not test.startswith("test_"):
        continue
    func = getattr(tests, test)
    if callable(func):
        print(f"Running {test}...")
        func()
        print(f"{test} completed.\n\n\n")