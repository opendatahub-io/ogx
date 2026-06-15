# OGX Midstream CI (ODH-specific)

Workflows specific to the OpenDataHub midstream fork, not present upstream.

| Name | File | Purpose |
| ---- | ---- | ------- |
| Create or Update Release Branch | [odh-create-or-update-release-branch.yml](odh-create-or-update-release-branch.yml) | Create or update release-${{ inputs.product_version }} from tag ${{ inputs.tag }} |
| Create release tag | [odh-create-tag.yml](odh-create-tag.yml) | Create tag from version in pyproject.toml |
| Dispatch Version Update to ODH Distribution | [odh-dispatch-version-update-to-odh-distribution.yml](odh-dispatch-version-update-to-odh-distribution.yml) | Dispatch version update to llama-stack-distribution (${{ github.ref_name }}) |
