from typing import Callable

import pytest

from src.core.hydrodynamics.hydrodynamic_protocol import HydrodynamicProtocol


class TestHydrodynamicProtocol:
    """
    Test fixture which is only meant to verify that all properties and methods
    in the protocol raise a NotImplementedError as they are meant to be defined
    in the binded classes through stric typing (e.g.: testClass: HydrodynamicProtocol)
    """

    def test_init_hydrodynamicprotocol(self):
        """
        A protocol cannot be instantiated.
        """
        with pytest.raises(TypeError) as e_err:
            HydrodynamicProtocol()

        assert str(e_err.value) == "Protocols cannot be instantiated"

    def test_protocol_as_type(self):
        """
        A protocol is binded by strict typing.
        """

        class TestBaseClass:
            @property
            def a_property(self):
                return None

        class TestClass(TestBaseClass):
            @property
            def working_dir(self):
                return None

            @property
            def config_file(self):
                return None

            @property
            def definition_file(self):
                return None

            @property
            def settings(self):
                return None

            @property
            def space(self):
                return None

            @property
            def water_depth(self):
                return None

            @property
            def x_coordinates(self):
                return None

            @property
            def y_coordinates(self):
                return None

            @property
            def xy_coordinates(self):
                return None

            def initiate(self):
                raise NotImplementedError

            def update(self):
                raise NotImplementedError

            def finalise(self):
                raise NotImplementedError

        unbinded_class: HydrodynamicProtocol = TestBaseClass()
        assert isinstance(unbinded_class, HydrodynamicProtocol) is False
        binded_class: HydrodynamicProtocol = TestClass()
        assert isinstance(binded_class, HydrodynamicProtocol)

    def test_inheritance_aint_everything(self):
        """
        A protocol-based model needs to implement its own definitions.
        """

        class TestDerived(HydrodynamicProtocol):
            pass

        test_derived = TestDerived()

        def verify_prop_raises(prop_name: str):
            with pytest.raises(NotImplementedError):
                getattr(test_derived, prop_name)
                pytest.fail(f"Property {prop_name} was supposed to raise.")

        def verify_method_raises(method_name: str, **kwargs):
            with pytest.raises(NotImplementedError):
                if kwargs:
                    getattr(test_derived, method_name)(**kwargs)
                else:
                    getattr(test_derived, method_name)()
                pytest.fail(f"Method {method_name} was supposed to raise.")

        props_to_test = [
            "working_dir",
            "config_file",
            "definition_file",
            "settings",
            "x_coordinates",
            "y_coordinates",
            "xy_coordinates",
            "water_depth",
            "space",
        ]
        list(map(verify_prop_raises, props_to_test))
        verify_method_raises("initiate")
        verify_method_raises("update", coral=None, stormcat=None)
        verify_method_raises("finalise")
