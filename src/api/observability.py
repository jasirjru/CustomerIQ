"""Low-cardinality, payload-free operational metrics for the inference API."""

from collections import Counter
from threading import Lock


HTTP_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
SCORE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
KNOWN_ROUTES = frozenset({
    "/", "/assets", "/docs", "/health", "/input-schema", "/metrics",
    "/model-info", "/openapi.json", "/predict", "/predict-batch",
})


def canonical_route(scope) -> str:
    route = getattr(scope.get("route"), "path", None)
    if route in KNOWN_ROUTES:
        return route
    path = scope.get("path", "")
    if path.startswith("/assets/"):
        return "/assets"
    return path if path in KNOWN_ROUTES else "unmatched"


def _labels(**labels) -> str:
    escaped = []
    for name, value in sorted(labels.items()):
        safe = str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
        escaped.append(f'{name}="{safe}"')
    return "{" + ",".join(escaped) + "}" if escaped else ""


class MetricsRegistry:
    """Thread-safe in-process counters suitable for one-worker deployments."""

    def __init__(self):
        self._lock = Lock()
        self.http_requests = Counter()
        self.http_duration_sum = Counter()
        self.http_duration_buckets = Counter()
        self.inference_requests = Counter()
        self.inference_duration_sum = Counter()
        self.predictions = Counter()
        self.score_count = 0
        self.score_sum = 0.0
        self.score_buckets = Counter()
        self.validation_errors = Counter()

    def record_http(self, method: str, route: str, status: int, duration_seconds: float):
        key = (method, route, str(status))
        with self._lock:
            self.http_requests[key] += 1
            self.http_duration_sum[(method, route)] += duration_seconds
            for boundary in HTTP_DURATION_BUCKETS:
                if duration_seconds <= boundary:
                    self.http_duration_buckets[(method, route, boundary)] += 1

    def record_inference(self, endpoint: str, responses, duration_seconds: float):
        with self._lock:
            self.inference_requests[endpoint] += 1
            self.inference_duration_sum[endpoint] += duration_seconds
            for response in responses:
                score = float(response.churn_probability)
                self.predictions[str(response.churn_prediction)] += 1
                self.score_count += 1
                self.score_sum += score
                for boundary in SCORE_BUCKETS:
                    if score <= boundary:
                        self.score_buckets[boundary] += 1

    def record_validation(self, endpoint: str, errors):
        # Validation type is defined by Pydantic, while endpoint is a fixed route.
        # Do not label with attacker-controlled field names or input values.
        with self._lock:
            for error in errors:
                self.validation_errors[(endpoint, str(error.get("type", "unknown")))] += 1

    def render(self, model_id: str, schema_version: str) -> str:
        with self._lock:
            lines = [
                "# HELP customeriq_build_info Loaded model and schema metadata.",
                "# TYPE customeriq_build_info gauge",
                f"customeriq_build_info{_labels(model_id=model_id, schema_version=schema_version)} 1",
                "# HELP customeriq_http_requests_total HTTP responses by canonical route and status.",
                "# TYPE customeriq_http_requests_total counter",
            ]
            for (method, route, status), value in sorted(self.http_requests.items()):
                lines.append(f"customeriq_http_requests_total{_labels(method=method, route=route, status=status)} {value}")
            lines.extend([
                "# HELP customeriq_http_request_duration_seconds Request duration by canonical route.",
                "# TYPE customeriq_http_request_duration_seconds histogram",
            ])
            route_counts = Counter()
            for (method, route, _status), value in self.http_requests.items():
                route_counts[(method, route)] += value
            for (method, route, boundary), value in sorted(self.http_duration_buckets.items()):
                lines.append(f"customeriq_http_request_duration_seconds_bucket{_labels(method=method, route=route, le=boundary)} {value}")
            for (method, route), value in sorted(route_counts.items()):
                lines.append(f"customeriq_http_request_duration_seconds_bucket{_labels(method=method, route=route, le='+Inf')} {value}")
                lines.append(f"customeriq_http_request_duration_seconds_count{_labels(method=method, route=route)} {value}")
                lines.append(f"customeriq_http_request_duration_seconds_sum{_labels(method=method, route=route)} {self.http_duration_sum[(method, route)]:.9f}")
            lines.extend([
                "# HELP customeriq_inference_requests_total Inference requests by endpoint.",
                "# TYPE customeriq_inference_requests_total counter",
            ])
            for endpoint, value in sorted(self.inference_requests.items()):
                lines.append(f"customeriq_inference_requests_total{_labels(endpoint=endpoint)} {value}")
                lines.append(f"customeriq_inference_duration_seconds_sum{_labels(endpoint=endpoint)} {self.inference_duration_sum[endpoint]:.9f}")
            lines.extend([
                "# HELP customeriq_predictions_total Prediction decisions returned by the serving policy.",
                "# TYPE customeriq_predictions_total counter",
            ])
            for decision, value in sorted(self.predictions.items()):
                lines.append(f"customeriq_predictions_total{_labels(decision=decision)} {value}")
            lines.extend([
                "# HELP customeriq_churn_score Uncalibrated serving-score distribution.",
                "# TYPE customeriq_churn_score histogram",
            ])
            for boundary in SCORE_BUCKETS:
                lines.append(f"customeriq_churn_score_bucket{_labels(le=boundary)} {self.score_buckets[boundary]}")
            lines.append(f"customeriq_churn_score_bucket{_labels(le='+Inf')} {self.score_count}")
            lines.append(f"customeriq_churn_score_count {self.score_count}")
            lines.append(f"customeriq_churn_score_sum {self.score_sum:.17g}")
            lines.extend([
                "# HELP customeriq_validation_errors_total Rejected fields grouped by bounded error type.",
                "# TYPE customeriq_validation_errors_total counter",
            ])
            for (endpoint, error_type), value in sorted(self.validation_errors.items()):
                lines.append(f"customeriq_validation_errors_total{_labels(endpoint=endpoint, error_type=error_type)} {value}")
            return "\n".join(lines) + "\n"
