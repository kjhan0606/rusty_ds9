//! Source-catalog table model. Supports tab-separated catalogs and the
//! whitespace-separated SExtractor "ASCII_HEAD" format (column metadata in
//! `# n NAME comment` lines, then numeric rows).

use std::fs;
use std::io;
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct Catalog {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Copy)]
pub enum SortDir { Asc, Desc }

impl Catalog {
    pub fn len(&self) -> usize { self.rows.len() }
    pub fn is_empty(&self) -> bool { self.rows.is_empty() }

    pub fn from_tsv(input: &str) -> Self {
        let mut lines = input.lines();
        let columns = lines
            .next()
            .map(|l| l.split('\t').map(str::to_owned).collect())
            .unwrap_or_default();
        let rows = lines
            .map(|l| l.split('\t').map(str::to_owned).collect())
            .collect();
        Self { columns, rows }
    }

    /// Parse SExtractor's "ASCII_HEAD" output. Header lines look like:
    ///   `#   1 NUMBER       Running object number`
    ///   `#   2 X_IMAGE      Object position along x   [pixel]`
    /// Body rows are whitespace-separated values matching the column count.
    /// Vector columns (e.g. `MAG_APER(8)`) currently get one entry per scalar
    /// column — we just skip multi-column entries by inferring count from the
    /// data row width.
    pub fn from_sextractor(input: &str) -> Self {
        let mut columns = Vec::new();
        let mut rows: Vec<Vec<String>> = Vec::new();
        for raw in input.lines() {
            let line = raw.trim();
            if line.is_empty() { continue; }
            if let Some(rest) = line.strip_prefix('#') {
                let parts: Vec<&str> = rest.trim().splitn(3, char::is_whitespace).collect();
                if parts.len() >= 2 && parts[0].parse::<usize>().is_ok() {
                    columns.push(parts[1].to_string());
                }
                continue;
            }
            let cells: Vec<String> = line.split_whitespace().map(str::to_owned).collect();
            rows.push(cells);
        }
        // pad columns if SExtractor had vector columns we missed
        if let Some(first) = rows.first() {
            while columns.len() < first.len() {
                columns.push(format!("col{}", columns.len() + 1));
            }
        }
        Self { columns, rows }
    }

    /// Comma-separated values with a header row. Quoted fields ("…") are
    /// supported but only as a flat string — no embedded comma escaping beyond
    /// the surrounding quotes.
    pub fn from_csv(input: &str) -> Self {
        let split_csv = |line: &str| -> Vec<String> {
            let mut out = Vec::new();
            let mut cur = String::new();
            let mut in_q = false;
            for c in line.chars() {
                match c {
                    '"' => in_q = !in_q,
                    ',' if !in_q => { out.push(cur.trim().to_string()); cur.clear(); }
                    _ => cur.push(c),
                }
            }
            out.push(cur.trim().to_string());
            out
        };
        let mut lines = input.lines().filter(|l| !l.trim().is_empty());
        let columns = lines.next().map(split_csv).unwrap_or_default();
        let rows: Vec<Vec<String>> = lines.map(split_csv).collect();
        Self { columns, rows }
    }

    /// Minimal VOTable parser — extracts FIELD names and TR/TD rows. Ignores
    /// namespaces, attributes other than FIELD's `name=`, and any RESOURCE
    /// nesting. Good enough for SIMBAD / Vizier output we care about.
    pub fn from_votable(input: &str) -> Self {
        let mut columns = Vec::new();
        let mut rows: Vec<Vec<String>> = Vec::new();
        let mut cur_row: Vec<String> = Vec::new();
        let mut buf = input;
        // simple "find next < … >" loop — values live between TD open/close tags
        while let Some(open) = buf.find('<') {
            let after_lt = &buf[open + 1..];
            let close = match after_lt.find('>') { Some(c) => c, None => break };
            let tag_full = &after_lt[..close];
            let tag = tag_full.trim_start_matches('/').split_whitespace().next().unwrap_or("");
            let is_close = tag_full.starts_with('/');
            let body_start = open + 1 + close + 1;
            // find next tag to delimit body
            let next_lt = buf[body_start..].find('<').map(|p| body_start + p).unwrap_or(buf.len());
            let body = buf[body_start..next_lt].trim();

            let lc = tag.to_ascii_lowercase();
            match lc.as_str() {
                "field" if !is_close => {
                    // pull `name="…"` from tag attributes
                    let name = tag_full.split_whitespace().find_map(|tok| {
                        let t = tok.trim_end_matches('/');
                        let t = t.strip_prefix("name=")?;
                        Some(t.trim_matches('"').trim_matches('\'').to_string())
                    });
                    columns.push(name.unwrap_or_else(|| format!("col{}", columns.len() + 1)));
                }
                "td" if !is_close => {
                    // decode minimal entities and collect
                    let v = body.replace("&amp;", "&")
                        .replace("&lt;", "<").replace("&gt;", ">")
                        .replace("&quot;", "\"");
                    cur_row.push(v);
                }
                "tr" if is_close => {
                    if !cur_row.is_empty() {
                        rows.push(std::mem::take(&mut cur_row));
                    }
                }
                _ => {}
            }
            buf = &buf[next_lt..];
        }
        // pad missing column names
        if let Some(first) = rows.first() {
            while columns.len() < first.len() {
                columns.push(format!("col{}", columns.len() + 1));
            }
        }
        Self { columns, rows }
    }

    /// Auto-detect from the first ~256 bytes — VOTable XML, SExtractor, CSV,
    /// TSV, or bare whitespace.
    pub fn from_text_auto(input: &str) -> Self {
        let head: String = input.chars().take(256).collect();
        if head.contains("<VOTABLE") || head.contains("<votable") || head.contains("<TABLE") {
            return Self::from_votable(input);
        }
        let first = input.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
        if first.trim_start().starts_with('#') {
            Self::from_sextractor(input)
        } else if first.contains('\t') {
            Self::from_tsv(input)
        } else if first.contains(',') {
            Self::from_csv(input)
        } else {
            // bare whitespace-separated, no header — synthesize column names
            let rows: Vec<Vec<String>> = input.lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.split_whitespace().map(str::to_owned).collect())
                .collect();
            let n = rows.first().map(|r| r.len()).unwrap_or(0);
            let columns = (1..=n).map(|i| format!("col{i}")).collect();
            Self { columns, rows }
        }
    }

    pub fn from_path<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        Ok(Self::from_text_auto(&text))
    }

    pub fn col_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }

    /// Look up the standard SExtractor X_IMAGE / Y_IMAGE columns (or
    /// fall back to `X` / `Y`, then to `XWIN_IMAGE` / `YWIN_IMAGE`).
    pub fn xy_columns(&self) -> Option<(usize, usize)> {
        const X_NAMES: &[&str] = &["X_IMAGE", "XWIN_IMAGE", "X", "x", "XCEN", "x_image"];
        const Y_NAMES: &[&str] = &["Y_IMAGE", "YWIN_IMAGE", "Y", "y", "YCEN", "y_image"];
        let xi = X_NAMES.iter().find_map(|n| self.col_index(n))?;
        let yi = Y_NAMES.iter().find_map(|n| self.col_index(n))?;
        Some((xi, yi))
    }

    /// Iterate (x, y) pairs for the standard image-coordinate columns,
    /// dropping any row that fails to parse.
    pub fn xy_iter(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        let xy = self.xy_columns();
        self.rows.iter().filter_map(move |r| {
            let (xi, yi) = xy?;
            let x = r.get(xi)?.parse::<f64>().ok()?;
            let y = r.get(yi)?.parse::<f64>().ok()?;
            Some((x, y))
        })
    }

    pub fn sort_by(&mut self, col: usize, dir: SortDir) {
        if col >= self.columns.len() { return; }
        let numeric = self.rows.iter().all(|r|
            r.get(col).map(|v| v.trim().parse::<f64>().is_ok()).unwrap_or(true)
        );
        self.rows.sort_by(|a, b| {
            let av = a.get(col).map(String::as_str).unwrap_or("");
            let bv = b.get(col).map(String::as_str).unwrap_or("");
            let ord = if numeric {
                let an: f64 = av.trim().parse().unwrap_or(f64::NAN);
                let bn: f64 = bv.trim().parse().unwrap_or(f64::NAN);
                an.partial_cmp(&bn).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                av.cmp(bv)
            };
            match dir { SortDir::Asc => ord, SortDir::Desc => ord.reverse() }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_csv() {
        let txt = "X_IMAGE,Y_IMAGE,MAG\n100.5,200.3,18.5\n300.0,150.7,19.2\n";
        let cat = Catalog::from_text_auto(txt);
        assert_eq!(cat.columns, vec!["X_IMAGE","Y_IMAGE","MAG"]);
        assert_eq!(cat.len(), 2);
        let xy: Vec<_> = cat.xy_iter().collect();
        assert_eq!(xy.len(), 2);
        assert!((xy[0].0 - 100.5).abs() < 1e-9);
    }

    #[test]
    fn parse_votable() {
        let txt = r#"<?xml version="1.0"?>
<VOTABLE>
  <RESOURCE><TABLE>
    <FIELD name="X_IMAGE"/>
    <FIELD name="Y_IMAGE"/>
    <FIELD name="MAG"/>
    <DATA><TABLEDATA>
      <TR><TD>100.5</TD><TD>200.3</TD><TD>18.5</TD></TR>
      <TR><TD>300.0</TD><TD>150.7</TD><TD>19.2</TD></TR>
    </TABLEDATA></DATA>
  </TABLE></RESOURCE>
</VOTABLE>"#;
        let cat = Catalog::from_text_auto(txt);
        assert_eq!(cat.columns.len(), 3);
        assert_eq!(cat.len(), 2);
        let xy: Vec<_> = cat.xy_iter().collect();
        assert_eq!(xy.len(), 2);
        assert!((xy[1].1 - 150.7).abs() < 1e-6);
    }

    #[test]
    fn parse_sextractor() {
        let txt = "\
#   1 NUMBER         Running object number
#   2 X_IMAGE        Object position along x [pixel]
#   3 Y_IMAGE        Object position along y [pixel]
#   4 MAG_AUTO       Kron-like magnitude
   1   100.5  200.3  18.5
   2   300.0  150.7  19.2
";
        let cat = Catalog::from_sextractor(txt);
        assert_eq!(cat.columns, vec!["NUMBER","X_IMAGE","Y_IMAGE","MAG_AUTO"]);
        assert_eq!(cat.rows.len(), 2);
        let xy: Vec<_> = cat.xy_iter().collect();
        assert_eq!(xy.len(), 2);
        assert!((xy[0].0 - 100.5).abs() < 1e-9);
    }
}
