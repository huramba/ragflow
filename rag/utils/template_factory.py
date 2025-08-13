from jinja2 import UndefinedError, Environment


glob_env = Environment()


def jinja_format(template: str, **variables) -> str:
    template_str = glob_env.from_string(template)
    return template_str.render(**variables)