
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from textwrap import wrap

class ChartStyle:
    def __init__(
        self,
        font='DejaVu Sans',
        title_size=18,
        subtitle_size=12,
        label_size=14,
        xtick_size=12,
        ytick_size=12,
        legend_size=12,
        date_fmt='%Y-%m',
        nyears=5,
        grid=True,
        tight_layout=True,
        style='default',
        facecolor='#f8f8f8',         # Axes background (offwhite)
        figure_facecolor="lightgrey", # Figure background (outside axes)
        figsize=(10, 6)               # Default figure size (width, height in inches)
    ):
        self.font = font
        self.title_size = title_size
        self.subtitle_size = subtitle_size
        self.label_size = label_size
        self.xtick_size = xtick_size
        self.ytick_size = ytick_size
        self.legend_size = legend_size
        self.date_fmt = date_fmt
        self.nyears = nyears
        self.grid = grid
        self.tight_layout = tight_layout
        self.style = style
        self.facecolor = facecolor
        self.figure_facecolor = figure_facecolor
        self.figsize = figsize

    def apply(self, ax, title=None, xlabel=None, ylabel=None, legend=True, date_axis=False,
              subtitle=None, subtitle_size=None, **overrides):
        """
        Apply style to an Axes. Any ChartStyle attributes can be overridden by passing them
        as keyword arguments in `overrides` (e.g. title_size=20, facecolor='#fff', figsize=(8,4)).
        """
        # Helper to get overridden value or fallback to self
        def _get(attr):
            if attr in overrides and overrides[attr] is not None:
                return overrides[attr]
            return getattr(self, attr)

        style_name = _get('style')
        plt.style.use(style_name)
        plt.rcParams['font.family'] = _get('font')

        ax.set_facecolor(_get('facecolor'))  # Set axes background
        ax.figure.patch.set_facecolor(_get('figure_facecolor'))  # Set figure background

        # title + subtitle handling: place title slightly above and subtitle under it with smaller font
        eff_title_size = subtitle_size or overrides.get('title_size') or self.title_size
        if title:
            # place title a little higher to make room for subtitle
            ax.set_title(title, fontsize=_get('title_size'), y=1.03)

        if subtitle:
            # compute effective subtitle size
            eff_sub_size = subtitle_size or overrides.get('subtitle_size') or self.subtitle_size or max(8, int(_get('title_size') * 0.85))
            # place subtitle under title; use axes transform so it stays with the axes
            ax.text(
                0.5, 1, subtitle,
                transform=ax.transAxes,
                ha='center', va='bottom',
                fontsize=eff_sub_size,
                color='dimgray'
            )

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=_get('label_size'))
        else:
            ax.set_xlabel('')

        if ylabel:
            ax.set_ylabel(ylabel, fontsize=_get('label_size'))

        ax.tick_params(axis='x', labelsize=_get('xtick_size'))
        ax.tick_params(axis='y', labelsize=_get('ytick_size'))

        if legend and ax.get_legend():
            ax.legend(fontsize=_get('legend_size'))

        grid_setting = _get('grid')
        ax.grid(grid_setting, alpha=0.3 if grid_setting else 0)

        if date_axis:
            ax.xaxis.set_major_locator(mdates.YearLocator(_get('nyears')))
            ax.xaxis.set_major_formatter(mdates.DateFormatter(_get('date_fmt')))

        if _get('tight_layout'):
            plt.tight_layout()


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from textwrap import wrap



def _add_footnote(ax, footnote, style=None, style_overrides=None):
    if not footnote:
        return
    try:
        renderer = ax.figure.canvas.get_renderer()
        bbox = ax.get_window_extent(renderer=renderer)
        width_points = bbox.width * 72 / ax.figure.dpi
        wrap_width = max(10, int(width_points / 7))
    except Exception:
        # Fallback conservative wrap width if renderer not available (headless)
        wrap_width = 60

    wrapped_text = "\n".join(wrap(footnote, wrap_width))
    # determine fontsize
    if style_overrides and 'label_size' in style_overrides and style_overrides['label_size'] is not None:
        fontsize = style_overrides['label_size']
    elif style and hasattr(style, 'label_size'):
        fontsize = style.label_size
    else:
        fontsize = 10

    ax.annotate(
        wrapped_text,
        xy=(0, -0.13), xycoords='axes fraction',
        fontsize=fontsize,
        ha='left', va='top',
        color='dimgray'
    )

# new helper to apply y-axis formatting
def _apply_y_format(ax, fmt):
    """
    fmt may be:
      - None (no-op)
      - a matplotlib.ticker.Formatter instance
      - a format string like '{:.2f}' or a format spec '.2f'
      - a callable that takes a single numeric arg and returns a string
    """
    if not fmt:
        return
    try:
        if isinstance(fmt, mticker.Formatter):
            ax.yaxis.set_major_formatter(fmt)
            return
        if isinstance(fmt, str):
            if '{' in fmt and '}' in fmt:
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: fmt.format(v)))
            else:
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: format(v, fmt)))
            return
        if callable(fmt):
            # callable receives value (we ignore pos)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: fmt(v)))
    except Exception:
        # silently ignore formatting errors
        return


def plot_bar(
    x, y, ax=None, style: ChartStyle = None, title=None, subtitle=None, xlabel=None, ylabel=None,
    legend=True, footnote=None, annotate=True, grid=None, style_kwargs: dict = None, y_fmt=None, **kwargs
):
    """
    style_kwargs: optional dict to override ChartStyle attributes per-plot (e.g. {'title_size': 20})
    subtitle: optional subtitle placed under title
    y_fmt: optional Y-axis formatter (string, callable or matplotlib Formatter)
    Any additional kwargs are forwarded to ax.bar.
    """
    style_kwargs = style_kwargs or {}

    figsize = style_kwargs.get('figsize') if 'figsize' in style_kwargs else (style.figsize if style else None)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(x, y, **kwargs)

    # Use chart-specific grid setting if provided, else ChartStyle default
    grid_setting = grid if grid is not None else (style_kwargs.get('grid') if 'grid' in style_kwargs else (style.grid if style else True))

    if style:
        style.apply(ax, title, xlabel, ylabel, legend, subtitle=subtitle, **style_kwargs)
    else:
        # When no style object is provided, allow some overrides to still be applied
        if title:
            ax.set_title(title)
        if subtitle:
            ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha='center', va='bottom', fontsize=12, color='dimgray')
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

    # apply y-axis format (explicit param > style_kwargs)
    eff_y_fmt = y_fmt if y_fmt is not None else style_kwargs.get('y_fmt')
    _apply_y_format(ax, eff_y_fmt)

    ax.grid(grid_setting, alpha=0.3 if grid_setting else 0)
    _add_footnote(ax, footnote, style, style_kwargs)

    if annotate:
        # Determine annotate fontsize
        ann_fs = style_kwargs.get('label_size') if 'label_size' in style_kwargs else (style.label_size if style else 10)
        for bar, val in zip(bars, y):
            ax.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=ann_fs,
                color='black'
            )
    return ax


def plot_line(
    x, y, ax=None, style: ChartStyle = None, title=None, subtitle=None, xlabel=None, ylabel=None,
    legend=True, label=None, date_axis=False, footnote=None, grid=None, style_kwargs: dict = None,
    annotate_last=False, annotate_last_fmt=None, y_fmt=None, **kwargs
):
    """
    annotate_last: if True annotate the last (most recent) y value on the chart
    annotate_last_fmt: optional format string (e.g. '{:.2f}') or callable to format the value
    y_fmt: optional Y-axis formatter (string, callable or matplotlib Formatter)
    """
    style_kwargs = style_kwargs or {}
    figsize = style_kwargs.get('figsize') if 'figsize' in style_kwargs else (style.figsize if style else None)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y, label=label, **kwargs)

    if style:
        style.apply(ax, title, xlabel, ylabel, legend, date_axis=date_axis, subtitle=subtitle, **style_kwargs)
    else:
        if title:
            ax.set_title(title)
        if subtitle:
            ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha='center', va='bottom', fontsize=12, color='dimgray')
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

    # apply y-axis format (explicit param > style_kwargs)
    eff_y_fmt = y_fmt if y_fmt is not None else style_kwargs.get('y_fmt')
    _apply_y_format(ax, eff_y_fmt)

    # Override grid if specified
    grid_setting = grid if grid is not None else (style_kwargs.get('grid') if 'grid' in style_kwargs else (style.grid if style else True))
    ax.grid(grid_setting, alpha=0.3 if grid_setting else 0)

    if label:
        legend_fs = style_kwargs.get('legend_size') if 'legend_size' in style_kwargs else (style.legend_size if style else 12)
        ax.legend(fontsize=legend_fs)

    # annotate latest value if requested
    if annotate_last:
        try:
            # support pandas Series/Index (.iloc) and lists/numpy arrays
            if hasattr(x, 'iloc'):
                last_x = x.iloc[-1]
            else:
                last_x = x[-1]
            if hasattr(y, 'iloc'):
                last_y = y.iloc[-1]
            else:
                last_y = y[-1]
        except Exception:
            last_x = None
            last_y = None

        if last_x is not None and last_y is not None:
            # determine formatter for the annotated value
            if annotate_last_fmt is None:
                try:
                    text_val = f"{float(last_y):.2f}"
                except Exception:
                    text_val = str(last_y)
            elif callable(annotate_last_fmt):
                text_val = annotate_last_fmt(last_y)
            else:
                try:
                    text_val = annotate_last_fmt.format(last_y)
                except Exception:
                    text_val = str(last_y)

            ann_fs = style_kwargs.get('label_size') if 'label_size' in style_kwargs else (style.label_size if style else 10)

            ax.annotate(
                text_val,
                xy=(last_x, last_y),
                xytext=(8, 3),
                textcoords="offset points",
                ha='left',
                va='bottom',
                fontsize=ann_fs,
                color='black',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8)
            )

    _add_footnote(ax, footnote, style, style_kwargs)
    return ax


def plot_dual_axis(
    x, y1, y2, style: ChartStyle = None, title=None, subtitle=None, xlabel=None, ylabel1=None, ylabel2=None,
    label1=None, label2=None, legend=True, date_axis=False, footnote=None, style_kwargs: dict = None, y1_fmt=None, y2_fmt=None, **kwargs
):
    """
    y1_fmt / y2_fmt: optional formatters for left / right Y axes (string, callable or matplotlib Formatter)
    """
    style_kwargs = style_kwargs or {}
    figsize = style_kwargs.get('figsize') if 'figsize' in style_kwargs else (style.figsize if style else None)
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    ax1.plot(x, y1, label=label1, color=kwargs.get('color1', 'tab:blue'))
    ax2.plot(x, y2, label=label2, color=kwargs.get('color2', 'tab:orange'))

    if style:
        style.apply(ax1, title, xlabel, ylabel1, legend, date_axis=date_axis, subtitle=subtitle, **style_kwargs)
        # For the second axis we don't re-place the title/subtitle; we only apply axis-specific overrides
        style.apply(ax2, None, None, ylabel2, legend, date_axis=date_axis, **style_kwargs)
    else:
        if title:
            ax1.set_title(title)
        if subtitle:
            ax1.text(0.5, 1.01, subtitle, transform=ax1.transAxes, ha='center', va='bottom', fontsize=12, color='dimgray')
        if xlabel:
            ax1.set_xlabel(xlabel)
        if ylabel1:
            ax1.set_ylabel(ylabel1)
        if ylabel2:
            ax2.set_ylabel(ylabel2)

    # apply y-axis formatters (explicit params > style_kwargs)
    eff_y1 = y1_fmt if y1_fmt is not None else style_kwargs.get('y1_fmt', style_kwargs.get('y_fmt'))
    eff_y2 = y2_fmt if y2_fmt is not None else style_kwargs.get('y2_fmt', style_kwargs.get('y_fmt'))
    _apply_y_format(ax1, eff_y1)
    _apply_y_format(ax2, eff_y2)

    if label1:
        ax1.legend(loc='upper left', fontsize=style_kwargs.get('legend_size') if 'legend_size' in style_kwargs else (style.legend_size if style else 12))
    if label2:
        ax2.legend(loc='upper right', fontsize=style_kwargs.get('legend_size') if 'legend_size' in style_kwargs else (style.legend_size if style else 12))

    _add_footnote(ax1, footnote, style, style_kwargs)
    return ax1, ax2


def plot_dual_axis_line_bar(
    x, y_line, y_bar, style: ChartStyle = None, title=None, subtitle=None, xlabel=None,
    ylabel_line=None, ylabel_bar=None, label_line=None, label_bar=None, legend=True, date_axis=False,
    color_line='tab:blue', color_bar='tab:orange', bar_alpha=0.5, bar_width=20, footnote=None, style_kwargs: dict = None, y1_fmt=None, y2_fmt=None, **kwargs
):
    """
    y1_fmt / y2_fmt: optional formatters for left / right Y axes (string, callable or matplotlib Formatter)
    """
    style_kwargs = style_kwargs or {}
    figsize = style_kwargs.get('figsize') if 'figsize' in style_kwargs else (style.figsize if style else None)
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # Line on left axis
    ax1.plot(x, y_line, label=label_line, color=color_line, **kwargs)
    # Bar on right axis
    ax2.bar(x, y_bar, label=label_bar, color=color_bar, alpha=bar_alpha, width=bar_width)

    # Apply ChartStyle
    if style:
        style.apply(ax1, title, xlabel, ylabel_line, legend, date_axis=date_axis, subtitle=subtitle, **style_kwargs)
        style.apply(ax2, None, None, ylabel_bar, legend, date_axis=date_axis, **style_kwargs)
    else:
        if title:
            ax1.set_title(title)
        if subtitle:
            ax1.text(0.5, 1.01, subtitle, transform=ax1.transAxes, ha='center', va='bottom', fontsize=12, color='dimgray')
        if xlabel:
            ax1.set_xlabel(xlabel)
        if ylabel_line:
            ax1.set_ylabel(ylabel_line)
        if ylabel_bar:
            ax2.set_ylabel(ylabel_bar)

    # apply y-axis formatters (explicit params > style_kwargs)
    eff_y1 = y1_fmt if y1_fmt is not None else style_kwargs.get('y1_fmt', style_kwargs.get('y_fmt'))
    eff_y2 = y2_fmt if y2_fmt is not None else style_kwargs.get('y2_fmt', style_kwargs.get('y_fmt'))
    _apply_y_format(ax1, eff_y1)
    _apply_y_format(ax2, eff_y2)

    # Legends
    if label_line:
        ax1.legend(loc='upper left', fontsize=style_kwargs.get('legend_size') if 'legend_size' in style_kwargs else (style.legend_size if style else 12))
    if label_bar:
        ax2.legend(loc='upper right', fontsize=style_kwargs.get('legend_size') if 'legend_size' in style_kwargs else (style.legend_size if style else 12))

    _add_footnote(ax1, footnote, style, style_kwargs)
    return ax1, ax2
