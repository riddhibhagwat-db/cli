package air

import (
	"fmt"
	"strings"
)

// gpuType is a wire-facing accelerator type submitted to the training service.
// The string value is what the backend expects, so it must be preserved exactly.
// The number in the name is the partition count (e.g. GPU_8xH100 is 8 GPUs).
type gpuType string

const (
	gpuType1xA10  gpuType = "GPU_1xA10"
	gpuType8xH100 gpuType = "GPU_8xH100"
	gpuType1xH100 gpuType = "GPU_1xH100"
)

// gpuTypes lists every valid type. Used for validation error messages.
var gpuTypes = []gpuType{gpuType1xA10, gpuType1xH100, gpuType8xH100}

func validGPUTypesHint() string {
	names := make([]string, len(gpuTypes))
	for i, g := range gpuTypes {
		names[i] = string(g)
	}
	return "valid types are: " + strings.Join(names, ", ")
}

// parseGPUType resolves a YAML accelerator_type string to a gpuType. The match is
// exact: the server's lookup is case-sensitive, so a wrong casing is rejected
// rather than fixed up, which would hide a typo in the user's YAML.
func parseGPUType(value string) (gpuType, error) {
	switch gpuType(value) {
	case gpuType1xA10, gpuType8xH100, gpuType1xH100:
		return gpuType(value), nil
	}
	return "", fmt.Errorf("invalid GPU type %q: %s", value, validGPUTypesHint())
}

// gpusPerNode returns the per-node GPU count, which is the partition count from
// the name (GPU_1xH100 -> 1, GPU_8xH100 -> 8). num_accelerators must be a
// multiple of this.
func gpusPerNode(g gpuType) (int, error) {
	switch g {
	case gpuType1xA10, gpuType1xH100:
		return 1, nil
	case gpuType8xH100:
		return 8, nil
	}
	return 0, fmt.Errorf("invalid GPU type %q", string(g))
}
