pub type Var = usize;

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Linear {
    pub terms: Vec<(Var, i128)>,
    pub constant: i128,
}

impl Linear {
    pub fn constant(k: i128) -> Self {
        Self {
            terms: Vec::new(),
            constant: k,
        }
    }

    pub fn term(mut self, v: Var, c: i128) -> Self {
        self.add(v, c);
        self
    }

    pub fn add(&mut self, v: Var, c: i128) {
        match self.terms.iter_mut().find(|t| t.0 == v) {
            Some(t) => t.1 += c,
            None => self.terms.push((v, c)),
        }
    }

    pub fn plus(&self, other: &Linear, k: i128) -> Linear {
        let mut out = self.clone();
        for &(v, c) in &other.terms {
            out.add(v, c * k);
        }
        out.constant += other.constant * k;
        out
    }

    pub fn offset(mut self, k: i128) -> Self {
        self.constant += k;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Domain {
    Free,

    Within(Vec<(i128, i128)>),
}

impl Domain {
    pub fn between(lo: i128, hi: i128) -> Self {
        Self::Within(if lo <= hi { vec![(lo, hi)] } else { Vec::new() })
    }

    pub fn of_values(values: impl IntoIterator<Item = i128>) -> Self {
        let mut values: Vec<i128> = values.into_iter().collect();
        values.sort_unstable();
        values.dedup();
        let mut pieces: Vec<(i128, i128)> = Vec::new();
        for v in values {
            match pieces.last_mut() {
                Some(last) if last.1 + 1 == v => last.1 = v,
                _ => pieces.push((v, v)),
            }
        }
        Self::Within(pieces)
    }

    pub fn meet(&self, other: &Domain) -> Domain {
        match (self, other) {
            (Domain::Free, d) | (d, Domain::Free) => d.clone(),
            (Domain::Within(a), Domain::Within(b)) => {
                let mut out = Vec::new();
                let (mut i, mut j) = (0, 0);
                while i < a.len() && j < b.len() {
                    let (lo, hi) = (a[i].0.max(b[j].0), a[i].1.min(b[j].1));
                    if lo <= hi {
                        out.push((lo, hi));
                    }
                    if a[i].1 < b[j].1 {
                        i += 1;
                    } else {
                        j += 1;
                    }
                }
                Domain::Within(out)
            }
        }
    }

    pub fn hull(&self) -> Option<(Option<i128>, Option<i128>)> {
        match self {
            Domain::Free => Some((None, None)),
            Domain::Within(pieces) => Some((Some(pieces.first()?.0), Some(pieces.last()?.1))),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Clause {
    pub equal: Vec<Linear>,
    pub at_least: Vec<Linear>,
}

#[derive(Clone, Debug, Default)]
pub struct Problem {
    pub domains: Vec<Domain>,
    pub equal: Vec<Linear>,
    pub at_least: Vec<Linear>,

    pub either: Vec<Vec<Clause>>,
}

const STEPS: usize = 1 << 16;
const SPLINTERS: u128 = 1 << 20;
const ENUMERATED: i128 = 1 << 12;

fn narrowest(rows: &[Row], n: usize) -> Option<(Var, i128, i128)> {
    let mut best: Option<(i128, Var, i128, i128)> = None;
    for v in 0..n {
        let constant = |r: &Row| r.coef[v] != 0 && r.coef.iter().enumerate().all(|(w, &c)| w == v || c == 0);
        let lo = rows.iter().filter(|r| constant(r) && r.coef[v] > 0).map(|r| ceil_div(-r.constant, r.coef[v])).max();
        let hi = rows.iter().filter(|r| constant(r) && r.coef[v] < 0).map(|r| floor_div(r.constant, -r.coef[v])).min();
        let (Some(lo), Some(hi)) = (lo, hi) else {
            continue;
        };
        let width = hi - lo;
        if width <= ENUMERATED && best.is_none_or(|(w, ..)| width < w) {
            best = Some((width, v, lo, hi));
        }
    }
    best.map(|(_, v, lo, hi)| (v, lo, hi))
}

impl Problem {
    pub fn var(&mut self, domain: Domain) -> Var {
        self.domains.push(domain);
        self.domains.len() - 1
    }

    pub fn free(&mut self) -> Var {
        self.var(Domain::Free)
    }

    pub fn between(&mut self, lo: i128, hi: i128) -> Var {
        self.var(Domain::between(lo, hi))
    }

    pub fn equal(&mut self, e: Linear) {
        self.equal.push(e);
    }

    pub fn at_least(&mut self, e: Linear) {
        self.at_least.push(e);
    }

    pub fn at_most(&mut self, e: Linear, k: i128) {
        self.at_least(Linear::constant(k).plus(&e, -1));
    }

    pub fn bounds(&self, e: &Linear) -> (Option<i128>, Option<i128>) {
        let (mut low, mut high) = (Some(e.constant), Some(e.constant));
        for &(v, c) in &e.terms {
            if c == 0 {
                continue;
            }
            let (lo, hi) = match self.domains[v].hull() {
                Some(h) => h,
                None => return (Some(1), Some(0)),
            };
            let (a, b) = if c > 0 { (lo, hi) } else { (hi, lo) };
            low = low.zip(a).and_then(|(l, a)| l.checked_add(a.checked_mul(c)?));
            high = high.zip(b).and_then(|(h, b)| h.checked_add(b.checked_mul(c)?));
        }
        (low, high)
    }

    pub fn feasible(&self) -> Option<bool> {
        let Some((first, rest)) = self.either.split_first() else {
            let mut solver = Solver { steps: 0 };
            return solver.split(self, self.domains.clone());
        };
        let mut unknown = false;
        for option in first {
            let mut chosen = self.clone();
            chosen.either = rest.to_vec();
            chosen.equal.extend(option.equal.iter().cloned());
            chosen.at_least.extend(option.at_least.iter().cloned());
            match chosen.feasible() {
                Some(true) => return Some(true),
                Some(false) => {}
                None => unknown = true,
            }
        }
        if unknown {
            None
        } else {
            Some(false)
        }
    }
}

#[derive(Clone)]
struct Row {
    coef: Vec<i128>,
    constant: i128,
    equality: bool,
}

struct Solver {
    steps: usize,
}

fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn floor_div(a: i128, b: i128) -> i128 {
    a.div_euclid(b)
}

fn ceil_div(a: i128, b: i128) -> i128 {
    -floor_div(-a, b)
}

fn modhat(a: i128, m: i128) -> Option<i128> {
    let two_a = a.checked_mul(2)?;
    Some(a - m * floor_div(two_a.checked_add(m)?, 2 * m))
}

impl Solver {
    fn split(&mut self, p: &Problem, domains: Vec<Domain>) -> Option<bool> {
        let n = domains.len();
        let mut rows = Vec::new();
        for (v, d) in domains.iter().enumerate() {
            let Some((lo, hi)) = d.hull() else {
                return Some(false);
            };
            if let Some(lo) = lo {
                let mut coef = vec![0; n];
                coef[v] = 1;
                rows.push(Row { coef, constant: -lo, equality: false });
            }
            if let Some(hi) = hi {
                let mut coef = vec![0; n];
                coef[v] = -1;
                rows.push(Row { coef, constant: hi, equality: false });
            }
        }
        for (list, equality) in [(&p.equal, true), (&p.at_least, false)] {
            for e in list {
                let mut coef = vec![0; n];
                for &(v, c) in &e.terms {
                    coef[v] += c;
                }
                rows.push(Row { coef, constant: e.constant, equality });
            }
        }
        let hull = self.solve(rows, n)?;
        if !hull {
            return Some(false);
        }
        let Some(v) = domains
            .iter()
            .position(|d| matches!(d, Domain::Within(pieces) if pieces.len() > 1))
        else {
            return Some(true);
        };
        let Domain::Within(pieces) = &domains[v] else {
            unreachable!()
        };
        let mut unknown = false;
        for &piece in pieces {
            let mut narrowed = domains.clone();
            narrowed[v] = Domain::Within(vec![piece]);
            match self.split(p, narrowed) {
                Some(true) => return Some(true),
                Some(false) => {}
                None => unknown = true,
            }
        }
        if unknown {
            None
        } else {
            Some(false)
        }
    }

    fn solve(&mut self, mut rows: Vec<Row>, mut n: usize) -> Option<bool> {
        loop {
            self.steps += 1;
            if self.steps > STEPS {
                return None;
            }
            if !normalize(&mut rows) {
                return Some(false);
            }
            if let Some(k) = rows.iter().position(|r| r.equality) {
                let row = rows[k].clone();
                if !self.eliminate_equality(&mut rows, &row, &mut n)? {
                    return Some(false);
                }
                continue;
            }
            let Some((x, cost)) = self.choose(&rows, n) else {
                return Some(true);
            };
            if cost >= SPLINTERS {
                if let Some((v, lo, hi)) = narrowest(&rows, n) {
                    let mut unknown = false;
                    for k in lo..=hi {
                        let mut case = rows.clone();
                        let mut coef = vec![0; n];
                        coef[v] = 1;
                        case.push(Row { coef, constant: -k, equality: true });
                        match self.solve(case, n) {
                            Some(true) => return Some(true),
                            Some(false) => {}
                            None => unknown = true,
                        }
                    }
                    return if unknown { None } else { Some(false) };
                }
            }
            let (lowers, uppers): (Vec<Row>, Vec<Row>) = rows
                .iter()
                .filter(|r| r.coef[x] != 0)
                .cloned()
                .partition(|r| r.coef[x] > 0);
            let others: Vec<Row> = rows.iter().filter(|r| r.coef[x] == 0).cloned().collect();
            if lowers.is_empty() || uppers.is_empty() {
                rows = others;
                continue;
            }
            let constant_bounds = lowers
                .iter()
                .chain(&uppers)
                .all(|r| r.coef.iter().enumerate().all(|(v, &c)| v == x || c == 0));
            if constant_bounds {
                let low = lowers.iter().map(|r| ceil_div(-r.constant, r.coef[x])).max().unwrap();
                let high = uppers.iter().map(|r| floor_div(r.constant, -r.coef[x])).min().unwrap();
                if low > high {
                    return Some(false);
                }
                rows = others;
                continue;
            }
            let exact = lowers.iter().all(|r| r.coef[x] == 1) || uppers.iter().all(|r| r.coef[x] == -1);
            let mut shadow = others.clone();
            for l in &lowers {
                for u in &uppers {
                    shadow.push(combine(l, u, x, exact)?);
                }
            }
            if exact {
                rows = shadow;
                continue;
            }
            let mut unknown = false;
            match self.solve(shadow, n) {
                Some(true) => return Some(true),
                Some(false) => {}
                None => unknown = true,
            }
            let bmax = uppers.iter().map(|r| -r.coef[x]).max().unwrap();
            for l in &lowers {
                let a = l.coef[x];
                if a < 2 {
                    continue;
                }
                let top = floor_div(a.checked_mul(bmax)?.checked_sub(a)?.checked_sub(bmax)?, bmax);
                for i in 0..=top {
                    let mut splinter = rows.clone();
                    let mut eq = l.clone();
                    eq.constant = eq.constant.checked_sub(i)?;
                    eq.equality = true;
                    splinter.push(eq);
                    match self.solve(splinter, n) {
                        Some(true) => return Some(true),
                        Some(false) => {}
                        None => unknown = true,
                    }
                }
            }
            return if unknown { None } else { Some(false) };
        }
    }

    fn choose(&self, rows: &[Row], n: usize) -> Option<(Var, u128)> {
        let mut best: Option<(u128, Var)> = None;
        for x in 0..n {
            let lowers: Vec<i128> = rows.iter().filter(|r| r.coef[x] > 0).map(|r| r.coef[x]).collect();
            let uppers: Vec<i128> = rows.iter().filter(|r| r.coef[x] < 0).map(|r| -r.coef[x]).collect();
            if lowers.is_empty() && uppers.is_empty() {
                continue;
            }
            let cost: u128 = if lowers.is_empty() || uppers.is_empty() {
                0
            } else if lowers.iter().all(|&a| a == 1) || uppers.iter().all(|&b| b == 1) {
                1 + (lowers.len() * uppers.len()) as u128
            } else {
                let bmax = *uppers.iter().max().unwrap();
                let splinters: u128 = lowers
                    .iter()
                    .map(|&a| (a.saturating_mul(bmax) - a - bmax).max(0) as u128 / bmax as u128 + 1)
                    .sum();
                SPLINTERS | splinters.saturating_mul((lowers.len() * uppers.len()) as u128)
            };
            if best.is_none_or(|(c, _)| cost < c) {
                best = Some((cost, x));
            }
        }
        best.map(|(c, x)| (x, c))
    }

    fn eliminate_equality(&mut self, rows: &mut Vec<Row>, row: &Row, n: &mut usize) -> Option<bool> {
        let Some(k) = (0..*n).filter(|&v| row.coef[v] != 0).min_by_key(|&v| row.coef[v].abs()) else {
            return Some(row.constant == 0);
        };
        let a = row.coef[k];
        let sign = a.signum();
        let mut expr = Row {
            coef: vec![0; *n],
            constant: 0,
            equality: false,
        };
        if a.abs() == 1 {
            for v in 0..*n {
                if v != k {
                    expr.coef[v] = -sign * row.coef[v];
                }
            }
            expr.constant = -sign * row.constant;
        } else {
            let m = a.abs() + 1;
            for v in 0..*n {
                if v != k {
                    expr.coef[v] = sign * modhat(row.coef[v], m)?;
                }
            }
            expr.constant = sign * modhat(row.constant, m)?;
            expr.coef.push(-sign * m);
            *n += 1;
            for r in rows.iter_mut() {
                r.coef.push(0);
            }
        }
        for r in rows.iter_mut() {
            let c = r.coef[k];
            if c == 0 {
                continue;
            }
            for v in 0..*n {
                r.coef[v] = r.coef[v].checked_add(c.checked_mul(expr.coef[v])?)?;
            }
            r.coef[k] = 0;
            r.constant = r.constant.checked_add(c.checked_mul(expr.constant)?)?;
        }
        Some(true)
    }
}

fn normalize(rows: &mut Vec<Row>) -> bool {
    let mut keep = Vec::with_capacity(rows.len());
    for mut r in rows.drain(..) {
        let g = r.coef.iter().fold(0, |g, &c| gcd(g, c));
        if g == 0 {
            if if r.equality { r.constant != 0 } else { r.constant < 0 } {
                return false;
            }
            continue;
        }
        if g > 1 {
            if r.equality {
                if r.constant % g != 0 {
                    return false;
                }
                r.constant /= g;
            } else {
                r.constant = floor_div(r.constant, g);
            }
            for c in r.coef.iter_mut() {
                *c /= g;
            }
        }
        keep.push(r);
    }
    keep.sort_by(|a, b| (a.coef.as_slice(), a.constant).cmp(&(b.coef.as_slice(), b.constant)));
    keep.dedup_by(|a, b| a.coef == b.coef && a.constant == b.constant && a.equality == b.equality);
    *rows = keep;
    true
}

fn combine(l: &Row, u: &Row, x: Var, exact: bool) -> Option<Row> {
    let (a, b) = (l.coef[x], -u.coef[x]);
    let mut coef = Vec::with_capacity(l.coef.len());
    for v in 0..l.coef.len() {
        coef.push(b.checked_mul(l.coef[v])?.checked_add(a.checked_mul(u.coef[v])?)?);
    }
    coef[x] = 0;
    let mut constant = b.checked_mul(l.constant)?.checked_add(a.checked_mul(u.constant)?)?;
    if !exact {
        constant = constant.checked_sub((a - 1).checked_mul(b - 1)?)?;
    }
    Some(Row {
        coef,
        constant,
        equality: false,
    })
}
