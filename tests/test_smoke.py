from importlib import import_module


def test_package_imports():
    import forex_v1

    assert forex_v1.__doc__


def test_architecture_packages_are_importable():
    package_names = (
        "data",
        "features",
        "strategies",
        "models",
        "portfolio",
        "backtest",
        "evaluation",
        "experiments",
    )

    for package_name in package_names:
        import_module(f"forex_v1.{package_name}")


def test_target_weight_provider_contract_is_importable():
    from forex_v1.contracts import TargetWeightProvider

    assert TargetWeightProvider
