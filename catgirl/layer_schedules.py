"""CPU-only validation for optional per-layer intervention timing."""
import math


def normalize_layer_schedules(value, layer_count=32):
    """Return canonical string layer keys and complete validated schedules."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError('layer_schedules must be a mapping')
    result = {}
    required = {'early_tokens', 'prefill_scale', 'tail_scale'}
    for key, schedule in value.items():
        if isinstance(key, bool) or not isinstance(key, (int, str)):
            raise ValueError('layer_schedules keys must be layer numbers')
        try:
            layer = int(key)
        except ValueError:
            raise ValueError('layer_schedules keys must be layer numbers') from None
        if str(layer) != str(key) or not 0 <= layer < layer_count:
            raise ValueError('layer_schedules layer is outside the model or noncanonical')
        if not isinstance(schedule, dict) or set(schedule) != required:
            raise ValueError('Each layer schedule needs exactly early_tokens, prefill_scale, tail_scale')
        early = schedule['early_tokens']
        if early is not None and (type(early) is not int or early < 0):
            raise ValueError('Layer early_tokens must be null or a nonnegative integer')
        normalized = {'early_tokens': early}
        for field in ('prefill_scale', 'tail_scale'):
            scale = schedule[field]
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
                raise ValueError('Layer timing scales must be finite numbers')
            normalized[field] = float(scale)
        if str(layer) in result:
            raise ValueError('Duplicate layer schedule')
        result[str(layer)] = normalized
    return dict(sorted(result.items(), key=lambda pair: int(pair[0])))
