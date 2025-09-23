# Kinetix Documentation

This directory contains the Quarto-based documentation website for Kinetix, a Python library for reactive transport modeling with JAX and PyMC.

## Building the Documentation

### Prerequisites

- [Quarto](https://quarto.org/) (version 1.3 or later)
- Python environment with Kinetix and dependencies installed
- Jupyter kernel for Python

### Installation

1. Install Quarto following the [installation guide](https://quarto.org/docs/get-started/)

2. Install Python dependencies using pixi:
```bash
# Install base environment
pixi install

# Install docs feature for Quarto
pixi install --feature docs
```

3. The Jupyter kernel is automatically available through pixi

### Building the Site

From the project root directory:

```bash
# Preview during development (with docs feature)
pixi run --feature docs docs-preview

# Build static site
pixi run --feature docs docs-build

# Build and publish to GitHub Pages (if configured)
pixi run --feature docs docs-publish
```

Or manually from the `docs/` directory with pixi shell:

```bash
pixi shell --feature docs
cd docs
quarto preview  # or render, publish
```

### Development Workflow

1. **Edit content**: Modify `.qmd` files in the docs directory
2. **Preview changes**: Run `quarto preview` to see live updates
3. **Test notebooks**: Ensure example notebooks run correctly
4. **Build final site**: Run `quarto render` before committing

## Directory Structure

```
docs/
├── _quarto.yml          # Main configuration
├── index.qmd            # Homepage
├── installation.qmd     # Installation guide
├── quickstart.qmd       # Quick start tutorial
├── examples/            # Example notebooks and tutorials
│   ├── transport-model.qmd
│   └── transport-with-pymc.qmd
├── api/                 # API reference
│   └── index.qmd
├── styles.css           # Custom CSS
├── custom.scss          # Custom SCSS theme
└── README.md            # This file
```

## Content Guidelines

### Writing Style
- Use clear, concise language
- Include practical examples
- Explain concepts before diving into technical details
- Use consistent terminology throughout

### Code Examples
- Test all code examples to ensure they run
- Use meaningful variable names
- Include comments for complex operations
- Show both basic and advanced usage patterns

### Mathematical Notation
- Use LaTeX for mathematical expressions
- Define symbols and variables clearly
- Provide physical interpretation of equations

### Figures and Plots
- Include descriptive captions
- Use consistent styling and color schemes
- Ensure plots are readable at different sizes
- Provide alternative text for accessibility

## Configuration

### Quarto Configuration (`_quarto.yml`)
- Website metadata and navigation
- Execution settings for notebooks
- Output formats and styling
- Cross-reference and bibliography settings

### Styling
- `custom.scss`: SCSS theme customizations
- `styles.css`: Additional CSS rules
- Color scheme based on blue/purple palette
- Responsive design for mobile devices

## Deployment

The documentation can be deployed to various platforms:

### GitHub Pages
```bash
quarto publish gh-pages
```

### Netlify
```bash
quarto publish netlify
```

### Custom Server
Build static files and deploy the `_site/` directory:
```bash
quarto render
# Deploy _site/ directory to your server
```

## Troubleshooting

### Common Issues

**Jupyter kernel not found**:
```bash
# Ensure pixi environment is active and docs feature is installed
pixi install --feature docs
pixi shell --feature docs
```

**Notebook execution errors**:
- Check that all dependencies are installed
- Verify notebook runs in Jupyter independently
- Check Python environment and package versions

**Quarto rendering errors**:
- Update Quarto to latest version
- Check YAML front matter syntax
- Verify file paths and references

**Styling issues**:
- Clear browser cache
- Check CSS/SCSS syntax
- Verify Bootstrap compatibility

### Getting Help

1. Check the [Quarto documentation](https://quarto.org/docs/)
2. Review [Kinetix GitHub issues](https://github.com/your-org/kinetix/issues)
3. Ask questions in the project discussions

## Contributing

### Adding New Content
1. Create new `.qmd` files in appropriate directories
2. Update `_quarto.yml` navigation structure
3. Test content renders correctly
4. Update cross-references as needed

### Improving Examples
1. Add new example notebooks to `examples/`
2. Ensure notebooks are well-documented
3. Include real-world applications when possible
4. Test with different parameter values

### Updating API Documentation
1. Keep API docs synchronized with code changes
2. Include practical usage examples
3. Document parameter types and return values
4. Add cross-references between related functions

## Maintenance

### Regular Tasks
- Update examples when API changes
- Check external links for validity
- Update installation instructions
- Review and improve content clarity

### Version Updates
- Update version numbers in examples
- Check compatibility with new dependencies
- Update screenshots and outputs
- Verify all notebooks execute successfully

## License

Documentation content is released under the same license as the Kinetix project (MIT License).