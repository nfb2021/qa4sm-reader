def note(note_text):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f'\n\n{note_text}\n\n')
            return func(*args, **kwargs)
        return wrapper
    return decorator
